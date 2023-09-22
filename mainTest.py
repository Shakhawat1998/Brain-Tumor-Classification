import cv2
from keras.models import load_model
from PIL import Image
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# load model

# model=load_model('BrainTumor10Epochs.h5')

model=load_model('BrainTumor10EPochsCategorical.h5')

image=cv2.imread('C:\\Users\\ACER\\Downloads\\Compressed\\Brain Tumor Detection and Classification\\datasets\\pred\\pred0.jpg')

img=Image.fromarray(image)

img=img.resize((64,64))

img=np.array(img)

input_img=np.expand_dims(img,axis=0)

# print(img)

result=model.predict(input_img)
print(result)

class_labels = np.argmax(result, axis=-1)

print(class_labels)


