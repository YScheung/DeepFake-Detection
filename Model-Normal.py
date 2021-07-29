# To use CNN models for deepfake detection of normal images

from tensorflow.keras.models import model_from_json
from PIL import Image
import numpy as np
from mtcnn.mtcnn import MTCNN
import cv2
import numpy
from os import listdir


image_path = input("Path to folder: ")


face_detector = MTCNN()
json_file = open('mesonet_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("mesonet_model_normal_best_weights.h5")
print("Loaded model")



def get_faces(img_path):
    detector = MTCNN()
    facelist = []
    frame = cv2.imread(img_path)
    data = detector.detect_faces(frame)
    biggest = 0
    if data != []:
        for faces in data:
            box = faces['box']
            area = box[3] * box[2]
            if area > biggest:
                biggest = area
                bbox = box
        
        bbox[0] = 0 if bbox[0] < 0 else bbox[0]
        bbox[1] = 0 if bbox[1] < 0 else bbox[1]
        frame=frame[bbox[1] - bbox[3] // 12 : bbox[1] + bbox[3], bbox[0] - bbox[2] // 12: bbox[0] + bbox[2] + bbox[2] // 12]
        face_crop = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        PIL_image = Image.fromarray(face_crop)
        PIL_image = PIL_image.resize((150,150))
        face_crop = cv2.cvtColor(numpy.array(PIL_image), cv2.COLOR_RGB2BGR)  
        fc_height, fc_width = face_crop.shape[:2]

        # pad the face image to 256x256
        img_padded = np.zeros((150, 150, 3), dtype=np.uint8)
        img_padded[:fc_height, :fc_width, :] = face_crop.copy()
         
        # scale the pixels            
        img_scaled = np.asarray(img_padded, dtype=np.float64)
        img_scaled /= (np.std(img_scaled, keepdims=True) + 1e-6)
        
        facelist.append(img_scaled)
        return facelist




for i in listdir(image_path):
    try:
      print(i)
      face_set = []
      face_set.extend(get_faces(image_path + "/" + i))
      predictions = loaded_model.predict(x=np.array(face_set),batch_size=10,verbose=0)
      print(predictions)
    except Exception:
        print("Cant access photo")
