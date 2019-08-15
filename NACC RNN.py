# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 14:15:27 2019

@author: Montague
"""
import random

import pickle
import keras
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Input, GlobalMaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

import sklearn
import sklearn.metrics
from keras.layers import Input, Conv2D, Dense, concatenate


cogscores1 = pd.read_csv("/Users/Montague/Desktop/DStuff/CogScore1")
cogscores2 = pd.read_csv("/Users/Montague/Desktop/DStuff/CogScore2")
cogscores3 = pd.read_csv("/Users/Montague/Desktop/DStuff/CogScore3")
cogscores4 = pd.read_csv("/Users/Montague/Desktop/DStuff/CogScore4")

cogscores1 = np.array(cogscores1)
cogscores2 = np.array(cogscores2)
cogscores3 = np.array(cogscores3)
cogscores4 = np.array(cogscores4)


future_cog1 = []
tfuture_cog1 = []
naccid = cogscores1[0][1]
indays = cogscores1[0][36]
print(naccid)
print(indays)
for i in range(len(cogscores1)-1):
    if (cogscores1[i][1] != naccid):
        naccid = cogscores1[i][1]
        print(naccid)
        indays = cogscores1[i][36]
        print(indays)
    if (cogscores1[i+1][1] == naccid and i < 9000):
        train = (cogscores1[i,2:37])
        days = cogscores1[i, 36] - indays
        label = cogscores1[i+1,18:26]
        train = np.append(train, days)
        thing = np.concatenate((train, label))
        tfuture_cog1.append(thing)
    if (cogscores1[i+1][1] == naccid and i >= 9000):
        train = (cogscores1[i,2:37])
        days = cogscores1[i, 36] - indays
        label = cogscores1[i+1,18:26]
        train = np.append(train, days)
        thing = np.concatenate((train, label))
        future_cog1.append(thing) 
        
print("1")
print(len(future_cog1))
print(len(tfuture_cog1))
future_cog1 = np.array(future_cog1)
tfuture_cog1 = np.array(tfuture_cog1)

future_cog2 = []
tfuture_cog2 = []

naccid = cogscores2[0][1]
indays = cogscores2[0][34]
print(naccid)
print(indays)
for i in range(len(cogscores2)-1):
    if (cogscores2[i][1] != naccid):
        naccid = cogscores2[i][1]
        print(naccid)
        indays = cogscores2[i][34]
        print(indays)    
    if (cogscores2[i+1][1] == naccid and i < 11000):
        train = (cogscores2[i,2:35])
        days = cogscores2[i, 34] - indays
        label = cogscores2[i+1,18:26]
        train = np.append(train, days)
        thing = np.concatenate((train, label))
        tfuture_cog2.append(thing)
    if (cogscores2[i+1][1] == naccid and i >= 11000):
        train = (cogscores2[i,2:35])
        days = cogscores2[i, 34] - indays
        label = cogscores2[i+1,18:26]
        train = np.append(train, days)
        thing = np.concatenate((train, label))
        future_cog2.append(thing) 

print("2")
print(len(future_cog2))
print(len(tfuture_cog2))
future_cog2 = np.array(future_cog2)
tfuture_cog2 = np.array(tfuture_cog2)

future_cog3 = []
tfuture_cog3 = []

naccid = cogscores3[0][1]
indays = cogscores3[0][29]
print(naccid)
print(indays)
for i in range(len(cogscores3)-1):
    if (cogscores3[i][1] != naccid):
        naccid = cogscores3[i][1]
        print(naccid)
        indays = cogscores3[i][29]
        print(indays)
    if (cogscores3[i+1][1] == naccid and i < 10500):
        train = (cogscores3[i,2:30])
        days = cogscores3[i, 29] - indays
        label = cogscores3[i+1,18:26]
        train = np.append(train, days)
        thing = np.concatenate((train, label))
        tfuture_cog3.append(thing)
    if (cogscores3[i+1][1] == naccid and i >= 10500):
        train = (cogscores3[i,2:30])
        days = cogscores3[i, 29] - indays
        label = cogscores3[i+1,18:26]
        train = np.append(train, days)
        thing = np.concatenate((train, label))
        future_cog3.append(thing) 
print("3")

print(len(future_cog3))
print(len(tfuture_cog3))
future_cog3 = np.array(future_cog3)
tfuture_cog3 = np.array(tfuture_cog3)

future_cog4 = []
tfuture_cog4 = []

naccid = cogscores4[0][1]
indays = cogscores4[0][27]
print(naccid)
print(indays)
for i in range(len(cogscores4)-1):
    if (cogscores4[i][1] != naccid):
        naccid = cogscores4[i][1]
        print(naccid)
        indays = cogscores4[i][27]
        print(indays)
    if (cogscores4[i+1][1] == naccid and i < 13804):
        train = (cogscores4[i,2:28])
        days = cogscores4[i, 27] - indays
        label = cogscores4[i+1,18:26]
        train = np.append(train, days)
        thing = np.concatenate((train, label))
        tfuture_cog4.append(thing)
    if (cogscores4[i+1][1] == naccid and i >= 13804):
        train = (cogscores4[i,2:28])
        days = cogscores4[i, 27] - indays
        label = cogscores4[i+1,18:26]
        train = np.append(train, days)
        thing = np.concatenate((train, label))
        future_cog4.append(thing) 
        
print("4")

print(len(future_cog4))
print(len(tfuture_cog4))
future_cog4 = np.array(future_cog4)
tfuture_cog4 = np.array(tfuture_cog4)

train_features1 = future_cog1[:,:35]
train_labels1 = future_cog1[:, 35:43]
print(train_labels1.shape)
print(train_features1.shape)
train_labels_mem1 = train_labels1[:, 0]
train_labels_ori1 = train_labels1[:, 1]
train_labels_jud1 = train_labels1[:, 2]
train_labels_com1 = train_labels1[:, 3]
train_labels_hom1 = train_labels1[:, 4]
train_labels_per1 = train_labels1[:, 5]
train_labels_cpt1 = train_labels1[:, 6]
train_labels_lan1 = train_labels1[:, 7]

ttrain_features1 = tfuture_cog1[:,:35]
ttrain_labels1 = tfuture_cog1[:, 35:43]
print(ttrain_labels1.shape)
print(ttrain_features1.shape)
ttrain_labels_mem1 = ttrain_labels1[:, 0]
ttrain_labels_ori1 = ttrain_labels1[:, 1]
ttrain_labels_jud1 = ttrain_labels1[:, 2]
ttrain_labels_com1 = ttrain_labels1[:, 3]
ttrain_labels_hom1 = ttrain_labels1[:, 4]
ttrain_labels_per1 = ttrain_labels1[:, 5]
ttrain_labels_cpt1 = ttrain_labels1[:, 6]
ttrain_labels_lan1 = ttrain_labels1[:, 7]

train_features2 = future_cog2[:,:33]
train_labels2 = future_cog2[:, 33:41]
print(train_labels2.shape)
print(train_features2.shape)
train_labels_mem2 = train_labels2[:, 0]
train_labels_ori2 = train_labels2[:, 1]
train_labels_jud2 = train_labels2[:, 2]
train_labels_com2 = train_labels2[:, 3]
train_labels_hom2 = train_labels2[:, 4]
train_labels_per2 = train_labels2[:, 5]
train_labels_cpt2 = train_labels2[:, 6]
train_labels_lan2 = train_labels2[:, 7]

ttrain_features2 = tfuture_cog2[:,:33]
ttrain_labels2 = tfuture_cog2[:, 33:41]
print(ttrain_labels2.shape)
print(ttrain_features2.shape)
ttrain_labels_mem2 = ttrain_labels2[:, 0]
ttrain_labels_ori2 = ttrain_labels2[:, 1]
ttrain_labels_jud2 = ttrain_labels2[:, 2]
ttrain_labels_com2 = ttrain_labels2[:, 3]
ttrain_labels_hom2 = ttrain_labels2[:, 4]
ttrain_labels_per2 = ttrain_labels2[:, 5]
ttrain_labels_cpt2 = ttrain_labels2[:, 6]
ttrain_labels_lan2 = ttrain_labels2[:, 7]

train_features3 = future_cog3[:,:28]
train_labels3 = future_cog3[:, 28:36]
print(train_labels3.shape)
print(train_features3.shape)
train_labels_mem3 = train_labels3[:, 0]
train_labels_ori3 = train_labels3[:, 1]
train_labels_jud3 = train_labels3[:, 2]
train_labels_com3 = train_labels3[:, 3]
train_labels_hom3 = train_labels3[:, 4]
train_labels_per3 = train_labels3[:, 5]
train_labels_cpt3 = train_labels3[:, 6]
train_labels_lan3 = train_labels3[:, 7]

ttrain_features3 = tfuture_cog3[:,:28]
ttrain_labels3 = tfuture_cog3[:, 28:36]
print(ttrain_labels3.shape)
print(ttrain_features3.shape)
ttrain_labels_mem3 = ttrain_labels3[:, 0]
ttrain_labels_ori3 = ttrain_labels3[:, 1]
ttrain_labels_jud3 = ttrain_labels3[:, 2]
ttrain_labels_com3 = ttrain_labels3[:, 3]
ttrain_labels_hom3 = ttrain_labels3[:, 4]
ttrain_labels_per3 = ttrain_labels3[:, 5]
ttrain_labels_cpt3 = ttrain_labels3[:, 6]
ttrain_labels_lan3 = ttrain_labels3[:, 7]

train_features4 = future_cog4[:,:26]
train_labels4 = future_cog4[:, 26:34]
print(train_labels4.shape)
print(train_features4.shape)
train_labels_mem4 = train_labels4[:, 0]
train_labels_ori4 = train_labels4[:, 1]
train_labels_jud4 = train_labels4[:, 2]
train_labels_com4 = train_labels4[:, 3]
train_labels_hom4 = train_labels4[:, 4]
train_labels_per4 = train_labels4[:, 5]
train_labels_cpt4 = train_labels4[:, 6]
train_labels_lan4 = train_labels4[:, 7]

ttrain_features4 = tfuture_cog4[:,:26]
ttrain_labels4 = tfuture_cog4[:, 26:34]
print(ttrain_labels4.shape)
print(ttrain_features4.shape)
ttrain_labels_mem4 = ttrain_labels4[:, 0]
ttrain_labels_ori4 = ttrain_labels4[:, 1]
ttrain_labels_jud4 = ttrain_labels4[:, 2]
ttrain_labels_com4 = ttrain_labels4[:, 3]
ttrain_labels_hom4 = ttrain_labels4[:, 4]
ttrain_labels_per4 = ttrain_labels4[:, 5]
ttrain_labels_cpt4 = ttrain_labels4[:, 6]
ttrain_labels_lan4 = ttrain_labels4[:, 7]

epochs = 10
batch_size = 64
learning_rate = .0001
concat = 1

in1 = Input(shape = train_features4[0].shape)
x1 = Dense(500, activation = 'relu')(in1)
x1 = keras.layers.Dropout(rate = .5)(x1)
x1 = Dense(100, activation = 'relu')(x1)
x1 = keras.layers.Dropout(rate = .5)(x1)
out1 = Dense(1, activation = 'sigmoid')(x1)

model1 = Model(inputs=[in1], outputs=[out1])
optimizer = Adam(lr=learning_rate)
model1.summary()
model1.compile(loss='binary_crossentropy',optimizer=optimizer)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,patience=1, min_lr=1e-20, verbose=1)

history = model1.fit(x = train_features4, y =train_labels_mem4,
                                  epochs=epochs,
                                  validation_data=([ttrain_features4],ttrain_labels_mem4),
                                  callbacks=[reduce_lr])

print("########################")
my_predict = model1.predict(ttrain_features4)
my_predict = my_predict.reshape(len(my_predict))
print(np.count_nonzero(my_predict))
print(sklearn.metrics.roc_auc_score(ttrain_labels_mem4, my_predict))
print(np.mean(np.absolute(np.subtract(ttrain_labels_mem4, my_predict))))