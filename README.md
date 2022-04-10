# Research on DeepFake detection 

## Introduction:
Multiple studies pointed out that the major distinctions between deepfake and real images are located at one’s eyes and lower face (mouth to chin). In this project, I would like to examine whether CNN models, namely mesonet, VGG19 and Resnet50 can classify images that only contain distinctive features of the object or more specifically, if deepfake classification could be done with only eyes and lower face of a person. 

In addition, I would like to test if models with less filters could perform deepfake detection on cropped images at a high accuracy. 


## Experimental Process:

### Data collection and preprcoessing: 
1) Extract frames from youtube videos 
2) Crop face photos from images obtained with the help of the [MTCNN module] (https://github.com/ipazc/mtcnn)
3) Crop eyes and lower face of people in face photos 


### Sample Image:
![last](https://user-images.githubusercontent.com/72407100/124893698-870e7380-e00d-11eb-8550-7bab57de5b97.jpg)







### Dataset prepared:  <br />
2000 real images  <br />
2000 deepfake images  <br />
80/20 train test split 


### Results: 
Unfortunately, CNN models performed worse when face images are cropped to only contain one’s eyes and lower face.

|     Model     |   Accuracy (Normal image)   |   Accuracy (Cropped image)  |
| ------------- | --------------------------- | --------------------------- |
|    Mesonet    |            75%              |             35%             |
|    VGG-19     |            70%              |             25%             |
|   Resnet-50   |            60%              |             30%             |

### Conclusion:
Below are the reasons that I believe led to the above expiremental results 
1) Excessive focus on distinctive features <br />
When applying CNN models to face photos that are cropped to only contain one’s eyes and lower face, filters may be trained to identify over detailed structures which is meaningless and non-differentiating

2) Ignored other facial features of a person <br />
Other facial features may still be needed in deepfake detection   

3) Primary assumption is false  <br />
The eyes and lower face of a person in face photo are not major distinctions between deepfake and real images
