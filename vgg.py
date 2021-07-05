import json
import pickle
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
import cv2, numpy as np



# SETTINGS

NUM_TRAIN_SAMPLES = 3200
NUM_VALIDATION_SAMPLES = 800
IMG_SIZE = 150

EPOCHS = 50
BATCH_SIZE = 16
TRAINING_STEPS = int(NUM_TRAIN_SAMPLES / BATCH_SIZE)
VALIDATION_STEPS = int(NUM_VALIDATION_SAMPLES / BATCH_SIZE)

LEARNING_RATE = 0.002

TRAIN_DATA_DIR = './dataset/train'
VALIDATION_DATA_DIR = './dataset/test'

print(os.listdir(TRAIN_DATA_DIR))


model = Sequential()
model.add(ZeroPadding2D((1,1),input_shape=(IMG_SIZE,IMG_SIZE,3)))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2), padding='same'))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2), padding='same'))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2), padding='same'))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2), padding='same'))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1))) 
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2), padding='same'))

model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='softmax'))



# TRAINING SETUP

run_model = model

run_model.compile(
    loss="binary_crossentropy",
    optimizer=Adam(),
    metrics=["accuracy"]
)

train_datagen = ImageDataGenerator(
    samplewise_center=False,
    samplewise_std_normalization=True,
   # zoom_range=0.2,
   # brightness_range=(.6, 1.4),
   # rotation_range=45,
   # width_shift_range=0.1,
   # height_shift_range=0.1,
   # horizontal_flip=True,
    data_format='channels_last',
)

validation_datagen = ImageDataGenerator(
    samplewise_center=False,
    samplewise_std_normalization=True,
    data_format='channels_last',
)

train_generator = train_datagen.flow_from_directory(
    directory=TRAIN_DATA_DIR,
    classes=['real', 'deepfake'],
    target_size=(IMG_SIZE, IMG_SIZE),
   batch_size=BATCH_SIZE,
    class_mode='binary')

validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DATA_DIR,
    classes=['real', 'deepfake'],
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary')



#early_stop = EarlyStopping(monitor='loss', min_delta=0.001, patience=3, mode='min', verbose=1)
checkpoint = ModelCheckpoint('vgg_model_normal_best_weights.h5', monitor='loss', verbose=1, save_best_only=True, mode='min', period=1)



H = run_model.fit_generator(
    generator=train_generator,
    steps_per_epoch=TRAINING_STEPS,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=VALIDATION_STEPS,
    verbose=2,
    workers=8,
    use_multiprocessing=False,
   # callbacks=[early_stop, checkpoint]
    callbacks = [checkpoint]
)



"""
with open('training_history', 'wb') as fp:
    pickle.dump(H, fp)
with open('training_history.json', 'w') as fp:
    json.dump(H.history, fp)
"""

model_json = model.to_json()
with open("VGG_model.json", "w") as json_file:
    json_file.write(model_json)


