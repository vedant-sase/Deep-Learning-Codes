import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import models,datasets,layers
import matplotlib.pyplot as plt
import matplotlib.image as mp
import sys
sys.stdout.reconfigure(encoding='utf-8')
(train_images,train_labels), (test_images,test_labels)=datasets.mnist.load_data("C:/Users/Vedant/Downloads/mnist.npz")
train_images=train_images/255
test_images= test_images/255
model=models.Sequential()
model.add(layers.Flatten(input_shape=(28, 28, 1)))
model.add(layers.Dense(32,activation="relu"))
model.add(layers.Dense(16,activation="relu"))
model.add(layers.Dense(10,activation="softmax"))    
model.compile(optimizer='adam',loss="sparse_categorical_crossentropy",metrics=
['accuracy'])
model.fit(train_images, train_labels, epochs=10,validation_data=(test_images,test_labels))
#For CIFAR
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import models,layers,datasets
(train_images, train_labels),(test_images,test_labels)=datasets.cifar10.load_d
ata()
train_images=train_images/255
test_images=test_images/255
model=models.Sequential()
model.add(layers.Flatten(input_shape=(32, 32, 3)))
model.add(layers.Dense(512,activation="relu"))
#model.add(layers.Dense(256,activation="relu"))
model.add(layers.Dense(128,activation="relu"))
#model.add(layers.Dense(64,activation="relu"))
model.add(layers.Dense(32,activation="relu"))
#model.add(layers.Dense(16,activation="relu"))
model.add(layers.Dense(10,activation="softmax"))
model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=
["accuracy"])
model.fit(train_images,train_labels,epochs=10,validation_data=(test_images,test_labels))
