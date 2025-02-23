# We wont use the 'nn.Module' but using some inspiration we'll develop our own API 
import torch
import torch.nn.functional as F
from engine import get_data, construct_vocab, construct_mapping, split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

NUM_GENERATIONS = 200
EMBEDDING_DIM = 10
BATCH_SIZE = 32
EPOCHS = 20000
CONTEXT_LENGTH  = 5
HIDDEN_DIM = 200
g = torch.Generator()
g.manual_seed(2147483647)

print("Current working directory:", os.getcwd())

#                -----------------------------
# setup a block | Linear -> BatchNorm -> Tanh | 
#                -----------------------------

class Linear:
    def __init__(self, fan_in, fan_out, bias= True):
        self.weight = torch.randn((fan_in, fan_out), generator=g) / fan_in**0.5
        self.bias = torch.zeros(fan_out) if bias else None 
    
    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out
    
    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])

class BatchNorm1d:
    def __init__(self, dimension, eps=1e-5, momentum=0.1):
        self.epsilon = eps
        self.momentum = momentum 
        self.training = True
        # following are the learnable parameters 
        self.gamma = torch.ones(dimension)
        self.beta = torch.zeros(dimension)
        # get a running "momentum avg"
        self.running_mean = torch.zeros(dimension)
        self.running_var = torch.ones(dimension)
    
    def __call__(self, x):
        if self.training:
            # compute the batch mean and batch variance
            xmean = x.mean(0, keepdim=True)
            xvar = x.var(0, keepdim=True)
        else:
            xmean = self.running_mean
            xvar = self.running_var
        
        # normalize to unit variance and zero mean 
        xhat = (x - xmean) / torch.sqrt(xvar + self.epsilon)
        # scale and shift 
        self.out = self.gamma * xhat + self.beta
        
        if self.training:
            with torch.no_grad():
                self.running_mean = (1-self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var = (1-self.momentum) * self.running_var + self.momentum * xvar
        return self.out
    
    def parameters(self):
        return [self.gamma, self.beta]

class Tanh:
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out
    
    def parameters(self):
        return []

class Model:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.num_embeddings = EMBEDDING_DIM
        self.num_hidden = HIDDEN_DIM
        self.C = None
        self.layers = None
        self.lossi = []
        self.update_to_data_ratio = []
        
    def __call__(self):
        self.C = torch.randn((self.vocab_size, self.num_embeddings), generator=g)
        self.layers = [
            Linear(self.num_embeddings * CONTEXT_LENGTH, self.num_hidden, bias=False), 
            BatchNorm1d(self.num_hidden),
            Tanh(),
            
            Linear(self.num_hidden, self.num_hidden, bias=False),
            BatchNorm1d(self.num_hidden),
            Tanh(),
            
            Linear(self.num_hidden, self.num_hidden, bias=False),
            BatchNorm1d(self.num_hidden),
            Tanh(),
            
            Linear(self.num_hidden, self.vocab_size, bias=False),
        ]
        return self.C, self.layers
    
    def train(self, Xtr, Ytr, Xdev, Ydev, Xte, Yte, parameters):
        print("Starting training...")
        self.lossi = [] 
        self.update_to_data_ratio = []  
        
        LEARNING_RATE = 0.1
        for epoch in range(EPOCHS):
            # minibatch construction
            index = torch.randint(0, Xtr.shape[0], (BATCH_SIZE,), generator=g)
            Xbatch = Xtr[index]
            Ybatch = Ytr[index]
            
            # forward pass 
            embeddings = self.C[Xbatch]
            x = embeddings.view(embeddings.shape[0], -1)
            for layer in self.layers:
                x = layer(x)
            
            # compute the loss 
            loss = F.cross_entropy(x, Ybatch)
            
            # backprop 
            for layer in self.layers:
                layer.out.retain_grad()
            for p in parameters:
                p.grad = None
            loss.backward() 
            
            # update 
            if epoch > EPOCHS * 0.8:
                LEARNING_RATE = 0.01
            for p in parameters:
                p.data -= LEARNING_RATE * p.grad
            
            # Track the stats with verification
            if epoch % 1000 == 0:
                print(f'{epoch:7d}/{EPOCHS:7d}: {loss.item():.4f}')
            self.lossi.append(loss.log10().item())
            
            # Add verification for update to data ratio
            with torch.no_grad():
                update_ratios = [((LEARNING_RATE * p.grad).std()/p.data.std()).log10().item() for p in parameters]
                self.update_to_data_ratio.append(update_ratios)
                if epoch % 1000 == 0:
                    print(f"Update ratios collected: {len(self.update_to_data_ratio)}")
    
    def get_track_stats(self):
        return self.lossi, self.update_to_data_ratio
    
    def debug_graphs(self, graph:str, parameters):
        print(f"\nAttempting to generate {graph}")
        try:
            if graph == "Activation Distribution":
                # Check if we have any data
                has_data = any(isinstance(layer, Tanh) and layer.out is not None for layer in self.layers[:-1])
                if not has_data:
                    print("No activation data found. Make sure to run training first.")
                    return
                plt.figure(figsize=(20,4))
                legends = []
                for i, layer in enumerate(self.layers[:-1]):
                    if isinstance(layer, Tanh):
                        t = layer.out
                        print('layer %d (%10s): mean %+.2f, std %.2f, saturated: %.2f%%' % (i, layer.__class__.__name__, t.mean(), t.std(), (t.abs() > 0.97).float().mean()*100))
                        hy, hx = torch.histogram(t, density=True)
                        plt.plot(hx[:-1].detach(), hy.detach())
                        legends.append(f'layer {i} ({layer.__class__.__name__})')
                plt.legend(legends)
                plt.title("Activation Distribution")
                plt.savefig('activation_distribution.png')
                print("Saved activation_distribution.png")
                plt.close()
            
            elif graph == "Gradient Distribution":
                # Check if we have any gradient data
                has_grads = any(isinstance(layer, Tanh) and layer.out.grad is not None for layer in self.layers[:-1])
                if not has_grads:
                    print("No gradient data found. Make sure to run training with backward pass.")
                    return
                plt.figure(figsize=(20,4))
                legends = []
                for i, layer in enumerate(self.layers[:-1]):
                    if isinstance(layer, Tanh):
                        t = layer.out.grad
                        print('layer %d (%10s): mean %+f, std %e' % (i, layer.__class__.__name__, t.mean(), t.std()))
                        hy, hx = torch.histogram(t, density=True)
                        plt.plot(hx[:-1].detach(), hy.detach())
                        legends.append(f'layer {i} ({layer.__class__.__name__}')
                plt.legend(legends);
                plt.title('gradient distribution')
                plt.savefig('gradient_distribution.png')
                plt.close()
            
            elif graph == "weight gradient distribution":
                # Check if parameters have gradients
                if not all(p.grad is not None for p in parameters):
                    print("Some parameters don't have gradients. Make sure training occurred.")
                    return
                plt.figure(figsize=(20, 4))
                legends = []
                for i,p in enumerate(parameters):
                    t = p.grad
                    if p.ndim == 2:
                        print('weight %10s | mean %+f | std %e | grad:data ratio %e' % (tuple(p.shape), t.mean(), t.std(), t.std() / p.std()))
                        hy, hx = torch.histogram(t, density=True)
                        plt.plot(hx[:-1].detach(), hy.detach())
                        legends.append(f'{i} {tuple(p.shape)}')
                plt.legend(legends)
                plt.title('weights gradient distribution')
                plt.savefig('weight_gradient_distribution.png')
                plt.close()
            
            elif graph == "update to data ratio":
                if not self.update_to_data_ratio:
                    print("No update to data ratio information collected during training.")
                    return
                plt.figure(figsize=(20, 4))
                legends = []
                for i,p in enumerate(parameters):
                    if p.ndim == 2:
                        plt.plot([self.update_to_data_ratio[j][i] for j in range(len(self.update_to_data_ratio))])
                        legends.append('param %d' % i)
                # update to data ratios should be ~1e-3
                plt.plot([0, len(self.update_to_data_ratio)], [-3, -3], 'k') 
                plt.legend(legends)
                plt.title('update to data ratio')
                plt.savefig('update_to_data_ratio.png')
                plt.close()         
            print(f"Successfully generated {graph}")
        except Exception as e:
            print(f"Error generating {graph}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def get_loss_stats(self,split, Xtr, Ytr, Xdev, Ydev, Xte, Yte):
        for layer in self.layers:
            layer.training = False
        with torch.no_grad():
              x,y = {
                    'train': (Xtr, Ytr),
                    'val': (Xdev, Ydev),
                    'test': (Xte, Yte),
                }[split]
              embeddings = self.C[x]
              x = embeddings.view(embeddings.shape[0], -1)
              for layer in self.layers:
                  x = layer(x)
              loss = F.cross_entropy(x, y)
              print(f'{split} loss: {loss.item():.4f}')
        layer.training = True

    def sample(self):
        # Set all BatchNorm layers to evaluation mode before sampling
        for layer in self.layers:
            if isinstance(layer, BatchNorm1d):
                layer.training = False
        
        # Remove final BatchNorm from the processing
        active_layers = [layer for layer in self.layers if not isinstance(layer, BatchNorm1d) or layer is not self.layers[-1]]
        
        samples = []
        for _ in range(NUM_GENERATIONS):
            out = []
            context = [0] * CONTEXT_LENGTH
            while True:
                emb = self.C[torch.tensor(context)]
                x = emb.view(1, -1)
                for layer in active_layers: 
                    x = layer(x)
                logits = x
                probs = F.softmax(logits, dim=1)
                ix = torch.multinomial(probs, num_samples=1).item()
                context = context[1:] + [ix]
                out.append(ix)
                if ix == 0:
                    break
            samples.append(''.join(vocab[i] for i in out))
        return '\n'.join(samples)
                
if __name__ == "__main__":
    corpus = get_data("https://raw.githubusercontent.com/danielmiessler/SecLists/refs/heads/master/Passwords/Common-Credentials/10-million-password-list-top-10000.txt", "passwords.txt")
    vocab, vocab_size = construct_vocab(corpus)
    element_to_index, index_to_element = construct_mapping(vocab)
    Xtr, Ytr, Xdev, Ydev, Xte, Yte = split(corpus, index_to_element, element_to_index)
    
    model = Model(len(element_to_index))
    C, layers = model()
    
    with torch.no_grad():
        for layer in layers[:-1]:
            if isinstance(layer, Linear):
                layer.weight *= 5/3
    
    parameters = [C] + [p for layer in layers for p in layer.parameters()]
    print(f"Total Model Parameters: {sum(p.nelement() for p in parameters)}")
    print("Setting requires_grad to True for all parameters")
    for p in parameters:
        p.requires_grad = True
        
    model.train(Xtr, Ytr, Xdev, Ydev, Xte, Yte, parameters)
    
    model.debug_graphs("Activation Distribution", parameters)
    model.debug_graphs("Gradient Distribution", parameters)
    model.debug_graphs("weight gradient distribution", parameters)
    model.debug_graphs("update to data ratio", parameters)
    
    model.get_loss_stats("train", Xtr, Ytr, Xdev, Ydev, Xte, Yte)
    model.get_loss_stats("val", Xtr, Ytr, Xdev, Ydev, Xte, Yte)
    model.get_loss_stats("test", Xtr, Ytr, Xdev, Ydev, Xte, Yte)
    
    print(model.sample())