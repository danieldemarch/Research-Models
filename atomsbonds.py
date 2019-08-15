# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 16:08:12 2019

@author: Montague
"""

import pickle
import keras
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Input, GlobalMaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
import json
import time
import numpy as np
import random

from dl_util import *
from ml_util import *
import pickle
import keras
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Input, GlobalMaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
import json
import time

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

import sklearn



print("Keras: %s"%keras.__version__)
x = np.load("f:/Users/Montague/Desktop/DStuff/testingx.npy")
y = np.load("f:/Users/Montague/Desktop/DStuff/testingy.npy")

posx = []
negx = []

posy = []
negy = []

for i in range(len(x)):
    if (y[i] == 1):
        posx.append(x[i])
        posy.append(1)
    else:
        negx.append(x[i])
        negy.append(0)
    
def cmol(mol, embed=15.0, res=0.5):
    dims = int(embed*2/res)
    cmol = Chem.Mol(mol.ToBinary())
    cmol.ComputeGasteigerCharges()
    AllChem.Compute2DCoords(cmol)
    coords = cmol.GetConformer(0).GetPositions()
    vect = np.zeros((dims,dims,4))
    #Bonds first
    for i,bond in enumerate(mol.GetBonds()):
        bondorder = bond.GetBondTypeAsDouble()
        bidx = bond.GetBeginAtomIdx()
        eidx = bond.GetEndAtomIdx()
        bcoords = coords[bidx]
        ecoords = coords[eidx]
        frac = np.linspace(0,1,int(1/res*2)) #
        for f in frac:
            c = (f*bcoords + (1-f)*ecoords)
            idx = int(round((c[0] + embed)/res))
            idy = int(round((c[1]+ embed)/res))
            #Save in the vector first channel
            if (idx >= dims):
                return 0
            if (idy >= dims):
                return 0
            if (idx < 0):
                return 0
            if (idy < 0):
                return 0
            vect[ idx , idy ,0] = bondorder
    #Atom Layers
    for i,atom in enumerate(cmol.GetAtoms()):
            idx = int(round((coords[i][0] + embed)/res))
            idy = int(round((coords[i][1]+ embed)/res))
            #Atomic number
            vect[ idx , idy, 1] = atom.GetAtomicNum()
            #Gasteiger Charges
            charge = atom.GetProp("_GasteigerCharge")
            vect[ idx , idy, 3] = charge
            #Hybridization
            hyptype = atom.GetHybridization().real
            vect[ idx , idy, 2] = hyptype
    return vect

def cmol2(mol, embed=15.0, res=0.5):
    dims = int(embed*2/res)
    cmol = Chem.Mol(mol.ToBinary())
    cmol.ComputeGasteigerCharges()
    AllChem.Compute2DCoords(cmol)
    coords = cmol.GetConformer(0).GetPositions()
    vect = np.zeros((dims,dims,1))
    #Bonds first
    for i,bond in enumerate(mol.GetBonds()):
        bondorder = bond.GetBondTypeAsDouble()
        bidx = bond.GetBeginAtomIdx()
        eidx = bond.GetEndAtomIdx()
        bcoords = coords[bidx]
        ecoords = coords[eidx]
        frac = np.linspace(0,1,int(1/res*2)) #
        for f in frac:
            c = (f*bcoords + (1-f)*ecoords)
            idx = int(round((c[0] + embed)/res))
            idy = int(round((c[1]+ embed)/res))
            #Save in the vector first channel
            if (idx >= dims):
                return 0
            if (idy >= dims):
                return 0
            if (idx < 0):
                return 0
            if (idy < 0):
                return 0
            vect[ idx , idy ,0] = bondorder
    #Atom Layers
    for i,atom in enumerate(cmol.GetAtoms()):
            idx = int(round((coords[i][0] + embed)/res))
            idy = int(round((coords[i][1]+ embed)/res))
            #Atomic number
            vect[ idx , idy, 0] = atom.GetAtomicNum()
    return vect

epochs = 20
batch_size = 128
learning_rate = .001
concat = 5

print(learning_rate, batch_size, epochs)

dposx = []
dnegx = []

dposy = []
dnegy = []

datax = []
datay = []

for i in range(len(posx)):
    mol = cmol2(posx[i])
    if isinstance(mol, int):
        print("dropped")
        print(i)
    else:
        dposx.append(mol)
        dposy.append(1)
print(len(dposx))

for i in range(len(negx)):
    mol = cmol2(negx[i])
    if isinstance(mol, int):
        print(i)
    else:
        dnegx.append(mol)
        dnegy.append(0)


postrx = []
posvax = []
postex = []

negtrx = []
negvax = []
negtex = []

postry = []
posvay = []
postey = []

negtry = []
negvay = []
negtey = []

for i in range(len(dposx)):
    if (i < len(dposx)/5):
        postex.append(dposx[i])
        postey.append(1)
    elif (i < len(dposx)*.3):
        posvax.append(dposx[i])
        posvay.append(1)
    else:
        postrx.append(dposx[i])
        postry.append(1)
    print(i)
        
for i in range(len(dnegx)):
    if (i < len(dnegx)/5):
        negtex.append(dnegx[i])
        negtey.append(0)
    elif (i < len(dnegx)*.3):
        negvax.append(dnegx[i])
        negvay.append(0)
    else:
        negtrx.append(dnegx[i])
        negtry.append(0)
    print(i)


X_train = np.array(negtrx+postrx*31)
X_val = np.array(negvax+posvax*31)
X_test = np.array(negtex+postex)

y_train_s = np.array(negtry+postry*31)
y_val_s = np.array(negvay+posvay*31)
y_test_s = np.array(negtey+postey)


task = "classification"

input_shape = X_train.shape[1:]
input_img = Input(shape=input_shape)

def Inception0(input):
    tower_1 = Conv2D(16, (1, 1), padding='same', activation='relu')(input)
    tower_1 = Conv2D(16, (3, 3), padding='same', activation='relu')(tower_1)

    tower_2 = Conv2D(16, (1, 1), padding='same', activation='relu')(input)
    tower_2 = Conv2D(16, (5, 5), padding='same', activation='relu')(tower_2)

    tower_3 = Conv2D(16, (1, 1), padding='same', activation='relu')(input)

    output = keras.layers.concatenate([tower_1, tower_2, tower_3], axis=-1)
    return output


def Inception(input):
    tower_1 = Conv2D(16, (1, 1), padding='same', activation='relu')(input)
    tower_1 = Conv2D(16, (3, 3), padding='same', activation='relu')(tower_1)

    tower_2 = Conv2D(16, (1, 1), padding='same', activation='relu')(input)
    tower_2 = Conv2D(16, (5, 5), padding='same', activation='relu')(tower_2)

    tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input)
    tower_3 = Conv2D(16, (1, 1), padding='same', activation='relu')(tower_3)

    output = keras.layers.concatenate([tower_1, tower_2, tower_3], axis=-1)
    return output


generator = ImageDataGenerator(rotation_range=180,
                               width_shift_range=0.1,height_shift_range=0.1,
                               fill_mode="constant",cval = 0,
                               horizontal_flip=True, vertical_flip=True,data_format='channels_last',)

x = Inception0(input_img)
x = Inception(x)
x = Inception(x)
od=int(x.shape[1])
x = MaxPooling2D(pool_size=(od,od), strides=(1,1))(x)
x = Flatten()(x)
x = Dense(100, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=input_img, outputs=output)
    #Concatenate for longer epochs
Xt = np.concatenate([X_train]*concat, axis=0)
yt = np.concatenate([y_train_s]*concat, axis=0)

g = generator.flow(Xt, yt, batch_size=batch_size, shuffle=True)
steps_per_epoch = 10000/batch_size

optimizer = Adam(lr=learning_rate)
model.summary()
print(len(X_train))
print(len(X_val))
print(len(X_test))
model.compile(loss='binary_crossentropy',optimizer=optimizer)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.25,patience=3, min_lr=1e-6, verbose=1)

start = time.time()
history = model.fit_generator(g,
                                  steps_per_epoch=len(Xt)//batch_size,
                                  epochs=epochs,
                                  validation_data=(X_val,y_val_s),
                                  callbacks=[reduce_lr])
stop = time.time()
time_elapsed = stop - start

#name = "chemception_"+dataset+"_epochs_"+str(epochs)+"_batch_"+str(batch_size)+"_learning_rate_"+str(learning_rate)
#model.save("%s.h5"%name)
#hist = history.history
#pickle.dump(hist, file("%s_history.pickle"%name,"w"))
print("########################")
#print("model and history saved",name)
print("########################")
y_predict = model.predict(X_test)
y_predict = y_predict.reshape(len(y_predict))
print(np.count_nonzero(y_predict))
print(sklearn.metrics.roc_auc_score(y_test_s, y_predict))
print(np.mean(np.absolute(np.subtract(y_test_s, y_predict))))
print(len(np.intersect1d(X_train, X_test)))
print(len(np.intersect1d(X_train, X_val)))
print(len(np.intersect1d(X_val, X_test)))
