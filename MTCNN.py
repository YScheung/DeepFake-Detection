from mtcnn import MTCNN
import cv2 
from PIL import Image
import matplotlib.pyplot as plt
from os import listdir
import os

from pkg_resources import NullProvider




def crop_image(image_path):
    try: 
      detector = MTCNN()
      img = cv2.imread(image_path)
      data = detector.detect_faces(img)
      biggest = 0
      if data != []:
        run = False
        for faces in data:
            box = faces['box']
            print(box)
            area = box[3] * box[2]
            if area > biggest and faces['confidence'] > 0.9:
                biggest = area
                bbox = box
                run = True
        
        if (biggest == 0):
            return (False,None)
        
        bbox[0] = 0 if bbox[0] < 0 else bbox[0]
        bbox[1] = 0 if bbox[1] < 0 else bbox[1]
        img=img[bbox[1] - bbox[3] // 12 : bbox[1] + bbox[3], bbox[0] - bbox[2] // 12: bbox[0] + bbox[2] + bbox[2] // 12]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return (True,img)

      else:
          return (False,None)
    
    except Exception:
      print("Failed")
      return (False,None)



count = 0
for i in listdir('/Users/yaushingjonathancheung/Desktop/realphotos'):
      if (count >= 0):
          print(i)
          img_path = '/Users/yaushingjonathancheung/Desktop/realphotos/' + i
          filename = str(count) + '.jpg'
          status,img=crop_image(img_path)
          path = '/Users/yaushingjonathancheung/Desktop/real-face'
          if status :
              cv2.imwrite(os.path.join(path,filename), img)
    
    
      count += 1
