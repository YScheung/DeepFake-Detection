# Research on DeepFake detection 

## Introduction:
From other studies, it is found out that major distinctions between deepfake and real images are located at one’s eyes and lower face (mouth to chin). 
Therefore, in this research project, I would like to examine whether different CNN models, namely mesonet, VGG19 and Resnet50 will perform better in deep fake detection when face photos are cropped to only contain identifying features compared to uncropped images. 

VGG19 is later removed from the research project as it performed poorly in deepfake detection 

## Experimental Process:

### Data collection and preprcoessing: 
1) Obtain frame images from youtube videos
2) Crop face photos from images obtained with the help of the [MTCNN module] (https://github.com/ipazc/mtcnn)
3) Crop eyes and lower face of people in face photos 

### Dataset prepared:  <br />
2000 real images  <br />
2000 deepfake images  <br />
80/20 train test split 


### Training process: 
Mesonet:  <br />
Epochs = 50, Batch size = 16, Learning rate = 0.002, Loss function = Binary crossentropy, Optimizer = Adam 




### Results: 

Unfortunately, CNN models performed worse when face images are cropped to only contain one’s eyes and lower face.


### Conclusion:
Below are the reasons that I believe led to the above expiremental results 
1) Excessive focus on distinctive features <br />
As face photos are cropped to only contain one’s eyes and lower face, parameters in determining whether a photo is deepfake or not are reduced. A lot more focus is placed on one’s eyes and lower face in deepfake detection which may lead to overfitting and harm the generalization of the model.   <br />
For example the model may give a different prediction when one’s eyes and lower face orientations differ slightly from the learned structure 

2) Ignored correlations between other facial features and eyes and lower face <br />
Other facial features may still be needed in deepfake detection through one’s eyes and lower face <br />
There may be a possibility that one’s eye placement is related to his face size. Yet in the proposed method, we have only focused on characteristics of one’s eyes and ignored his face size. This shows that the proposed method has assumed that there are no correlations between eyes, lower face and other facial features which may be false. 

3) Primary assumption is false <br />
The eyes and lower face of people in face photos are not major distinctions between deepfake and real photos 


### References:
https://arxiv.org/pdf/1809.00888.pdf
