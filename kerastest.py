import cv2
import glob
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from joblib import dump, load
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import layers
import os

data=[]
labels=[]
c=0
for item in glob.glob("Dataset/train/*"):
    c=c+1
    print(c)
    img = cv2.imread(item,cv2.IMREAD_GRAYSCALE) # read images with one channel grayscale
    r_img= cv2.resize(img,(128,128)) # resize to 128x128
    r_img = np.expand_dims(r_img, axis = -1)
    data.append(r_img) # add resized image to dataset list
    label =item.split("/")[2].split(".")[0]
    labels.append(label) #add image label to dataset list

#preprocess
le = LabelEncoder()
labels = le.fit_transform(labels)
labels = to_categorical(labels)

data = np.array(data)/255 # Normalize channel between 0 to 1

#split test and train randomly
x_train, x_test, y_train, y_test = train_test_split(data,labels,test_size=0.2)

#train
net= models.Sequential(
                        [
                            layers.Conv2D(32,(3,3),strides=(1,1),activation="relu",input_shape=(128,128,1)),
                            layers.Conv2D(32,(3,3),strides=(1,1),activation="relu"),
                            layers.BatchNormalization(),
                            layers.MaxPool2D((3,3)),
                            layers.Conv2D(64,(5,5),strides=(1,1),activation="relu"),
                            layers.Conv2D(64,(5,5),strides=(1,1),activation="relu"),
                            layers.BatchNormalization(),
                            layers.AvgPool2D((3,3)),
                            layers.Dropout(0.75),
                            layers.Flatten(),
                            layers.Dense(64,activation="relu"),
                            layers.Dense(16,activation="relu"),
                            layers.Dense(2,activation="softmax")
                        ]
                    )

print(net.summary())

net.compile(optimizer="SGD", loss="binary_crossentropy",metrics=["accuracy"])

H = net.fit(x_train,y_train,batch_size=32, epochs=24, validation_data=(x_test,y_test))

net.save("CatDogNew.h5")  # Save the model