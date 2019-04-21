import numpy as np
import cv2

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers.normalization import BatchNormalization
from keras import backend as K
import h5py

# dimensions of our images.
img_width, img_height = 48, 48


if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
              
model.load_weights('first_try2.h5')

# prevents openCL usage and unnecessary logging messages
cv2.ocl.setUseOpenCL(False)

# dictionary which assigns each label an emotion (alphabetical order)
emotion_dict = {0: "Happy", 1: "Sad"}


frame = cv2.imread('779.jpg',1) #7 #63
img_rows=48
img_cols=48
im = cv2.resize(frame,  (img_rows, img_cols)) 
#im.reshape((img_rows,img_cols))
print(im.shape) # (28,28)

batch = np.expand_dims(im,axis=0)
print(batch.shape) # (1, 28, 28)


prediction = model.predict(batch)
maxindex = int(np.argmax(prediction))
print('pred : ',emotion_dict[maxindex])
#cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

cv2.imshow('Video', cv2.resize(frame,(500,500),interpolation = cv2.INTER_CUBIC))
cv2.waitKey(0)


cv2.destroyAllWindows()