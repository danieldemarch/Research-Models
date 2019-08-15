# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 17:52:19 2019

@author: Montague
"""

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

from rdkit.Chem import MACCSkeys
from rdkit.Chem import DataStructs

from keras.layers import Input, Conv2D, Dense, concatenate



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

epochs = 2
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

mposx = []
mnegx = []

mposy = []
mnegy = []

matax = []
matay = []

fposx = []
fnegx = []

fposy = []
fnegy = []

fatax = []
fatay = []

for i in range(len(posx)):
    mol = cmol2(posx[i])
    mac = np.zeros(167)
    DataStructs.ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(posx[i]), mac)
    fin = np.zeros(2048)
    DataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(posx[i], radius=2), fin)

    if isinstance(mol, int):
        print("dropped")
        print(i)
    else:
        dposx.append(mol)
        dposy.append(1)
        mposx.append(mac)
        mposy.append(1)
        fposx.append(fin)
        fposy.append(1)
print(len(dposx))

for i in range(len(negx)):
    mol = cmol2(negx[i])
    mac = np.zeros(167)
    DataStructs.ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(negx[i]), mac)
    fin = np.zeros(2048)
    DataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(negx[i], radius=2), fin)
    if isinstance(mol, int):
        print(i)
    else:
        dnegx.append(mol)
        dnegy.append(0)
        mnegx.append(mac)
        mnegy.append(0)
        fnegx.append(fin)
        fnegy.append(0)

dpostrx = []
dposvax = []
dpostex = []

dnegtrx = []
dnegvax = []
dnegtex = []

dpostry = []
dposvay = []
dpostey = []

dnegtry = []
dnegvay = []
dnegtey = []

mpostrx = []
mposvax = []
mpostex = []

mnegtrx = []
mnegvax = []
mnegtex = []

mpostry = []
mposvay = []
mpostey = []

mnegtry = []
mnegvay = []
mnegtey = []

fpostrx = []
fposvax = []
fpostex = []

fnegtrx = []
fnegvax = []
fnegtex = []

fpostry = []
fposvay = []
fpostey = []

fnegtry = []
fnegvay = []
fnegtey = []

for i in range(len(mposx)):
    if (i < len(mposx)/5):
        mpostex.append(mposx[i])
        mpostey.append(1)
        fpostex.append(fposx[i])
        fpostey.append(1)
        dpostex.append(dposx[i])
        dpostey.append(1)
    elif (i < len(mposx)*.3):
        mposvax.append(mposx[i])
        mposvay.append(1)
        fposvax.append(fposx[i])
        fposvay.append(1)
        dposvax.append(dposx[i])
        dposvay.append(1)
    else:
        mpostrx.append(mposx[i])
        mpostry.append(1)
        fpostrx.append(fposx[i])
        fpostry.append(1)
        dpostrx.append(dposx[i])
        dpostry.append(1)
    print(i)
        
for i in range(len(mnegx)):
    if (i < len(mnegx)/5):
        mnegtex.append(mnegx[i])
        mnegtey.append(0)
        fnegtex.append(fnegx[i])
        fnegtey.append(0)
        dnegtex.append(dnegx[i])
        dnegtey.append(0)
    elif (i < len(mnegx)*.3):
        mnegvax.append(mnegx[i])
        mnegvay.append(0)
        fnegvax.append(fnegx[i])
        fnegvay.append(0)
        dnegvax.append(dnegx[i])
        dnegvay.append(0)
    else:
        mnegtrx.append(mnegx[i])
        mnegtry.append(0)
        fnegtrx.append(fnegx[i])
        fnegtry.append(0)
        dnegtrx.append(dnegx[i])
        dnegtry.append(0)
    print(i)


dX_train = np.array(dnegtrx+dpostrx*31)
dX_val = np.array(dnegvax+dposvax*31)
dX_test = np.array(dnegtex+dpostex)

dy_train_s = np.array(dnegtry+dpostry*31)
dy_val_s = np.array(dnegvay+dposvay*31)
dy_test_s = np.array(dnegtey+dpostey)

mX_train = np.array(mnegtrx+mpostrx*31)
mX_val = np.array(mnegvax+mposvax*31)
mX_test = np.array(mnegtex+mpostex)

my_train_s = np.array(mnegtry+mpostry*31)
my_val_s = np.array(mnegvay+mposvay*31)
my_test_s = np.array(mnegtey+mpostey)

fX_train = np.array(fnegtrx+fpostrx*31)
fX_val = np.array(fnegvax+fposvax*31)
fX_test = np.array(fnegtex+fpostex)

fy_train_s = np.array(fnegtry+fpostry*31)
fy_val_s = np.array(fnegvay+fposvay*31)
fy_test_s = np.array(fnegtey+fpostey)

input_shape = dX_train.shape[1:]
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

xd = Inception0(input_img)
xd = Inception(xd)
xd = Inception(xd)
od=int(xd.shape[1])
xd = MaxPooling2D(pool_size=(od,od), strides=(1,1))(xd)
xd = Flatten()(xd)
xd = Dense(100, activation='relu')(xd)
outd = Dense(1, activation='sigmoid')(xd)

inm = Input(shape = (167,))
xm = Dense(167, activation = 'relu')(inm)
xm = Dense(167, activation = 'relu')(xm)
xm = Dense(167, activation = 'relu')(xm)
outm = Dense(1, activation = 'sigmoid')(xm)


inf = Input(shape = (2048,))
xf = Dense(167, activation = 'relu')(inf)
xf = Dense(167, activation = 'relu')(xf)
xf = Dense(167, activation = 'relu')(xf)
outf = Dense(1, activation = 'sigmoid')(xf)

x = concatenate([outm, outf, outd])
out = Dense(1, activation = 'sigmoid')(x)

model = Model(inputs=[inm, inf, input_img], outputs=[out])

#Concatenate for longer epochs
fXt = np.concatenate([fX_train]*concat, axis=0)
fyt = np.concatenate([fy_train_s]*concat, axis=0)  

#Concatenate for longer epochs
dXt = np.concatenate([dX_train]*concat, axis=0)
dyt = np.concatenate([dy_train_s]*concat, axis=0)    

#Concatenate for longer epochs
mXt = np.concatenate([mX_train]*concat, axis=0)
myt = np.concatenate([my_train_s]*concat, axis=0)

steps_per_epoch = 10000/batch_size

optimizer = Adam(lr=learning_rate)
model.summary()
print(len(mX_train))
print(len(mX_val))
print(len(mX_test))
model.compile(loss='binary_crossentropy',optimizer=optimizer)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.25,patience=3, min_lr=1e-6, verbose=1)

start = time.time()
history = model.fit(x = [mXt, fXt, dXt], y = myt,
                                  epochs=epochs,
                                  validation_data=([mX_val, fX_val, dX_val],my_val_s),
                                  callbacks=[reduce_lr])
stop = time.time()
time_elapsed = stop - start

print("########################")
my_predict = model.predict([mX_test, fX_test, dX_test])
my_predict = my_predict.reshape(len(my_predict))
print(np.count_nonzero(my_predict))
print(sklearn.metrics.roc_auc_score(my_test_s, my_predict))
print(np.mean(np.absolute(np.subtract(my_test_s, my_predict))))