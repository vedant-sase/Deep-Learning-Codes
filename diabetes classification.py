import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys
sys.stdout.reconfigure(encoding='utf-8')

df=pd.read_csv("D:/archive (7)/diabetes.csv")
x=df.iloc[:,0:8]
y=df["Outcome"]
obj=StandardScaler()
x_=obj.fit_transform(x)
Xtrain,Xtest,Ytrain,Ytest=train_test_split(x_,y,test_size=0.1)
model=models.Sequential()
model.add(layers.Dense(100,activation="relu"))
#model.add(layers.Dense(75,activation="relu"))
model.add(layers.Dense(50,activation="relu"))
#model.add(layers.Dense(25,activation="relu"))
model.add(layers.Dense(12,activation="relu"))
model.add(layers.Dense(8,activation="relu"))
model.add(layers.Dense(1,activation="sigmoid"))
model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"]
,)
history=model.fit(Xtrain,Ytrain,epochs=50, validation_data=(Xtest,Ytest))
result=model.evaluate(Xtest,Ytest)
import matplotlib.pyplot as plt
plt.plot(history.history['loss'], label='loss')
#plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.ylim([0, 0.8])
plt.legend(loc='lower right')
test_loss, test_acc = model.evaluate(Xtest, Ytest, verbose=2)
plt.ylim([0.6,1])
plt.plot(history.history['accuracy'], label = 'accuracy')
