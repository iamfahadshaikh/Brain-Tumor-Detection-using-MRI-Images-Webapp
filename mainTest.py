import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

model=load_model('BrainTumor10EpochsCategorical.h5')

# image=cv2.imread('C:\\Users\\fahad\\OneDrive\\Desktop\\T.E MINI PROJECT\\N1.jpg')
image=cv2.imread('C:\\Users\\fahad\\OneDrive\\Desktop\\T.E MINI PROJECT\\Y1.jpg')

img=Image.fromarray(image)

img=img.resize((64,64))

img=np.array(img)

input_img=np.expand_dims(img, axis=0)

# result=model.predict_classes(input_img)
predict_x=model.predict(input_img) 
classes_x=np.argmax(predict_x,axis=1)
print(classes_x)




