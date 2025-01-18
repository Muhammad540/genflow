# genflow
An auto-regressive video generation model 

* resize image to a lower resolution 
* convert to gray scale 
* combine dataset from a number of videos into one dataset. 
* vocabulary can be value of each pixel value the model sees 
[80 , 234 
             -> for gray scale the pixel value is 0 to 255 -> discretize the pixels into bins  
 12 , 128]  

* range 0-255 discretized into 64 bins and each pixel assigned a particular bin value 
* at inference i will decode the model output 
*  Pixels within a frame and across frames are related, unlike independent characters in a sequence.