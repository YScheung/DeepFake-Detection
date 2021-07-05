# Research on DeepFake detection 

## Introduction:
From other studies, it is found out that major distinctions between deepfake and real images are located at one’s eyes and lower face (mouth to chin). 
Therefore, in this research project, I would like to examine whether different CNN models, namely mesonet, VGG19 and Resnet50 will perform better in deep fake detection when face photos are cropped to only contain identifying features compared to uncropped images. 

## Experimental Process:

### Data collection and preprcoessing: 
1) Obtain frame images from youtube videos
2) Crop face photos from images obtained with the help of the [MTCNN module] (https://github.com/ipazc/mtcnn)
3) Crop eyes and lower face of people in face photos 

Dataset prepared:  <br />
2000 real images  <br />
2000 deepfake images  <br />
80/20 train test split 
