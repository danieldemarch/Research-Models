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
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem import AllChem


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

epochs = 100
batch_size = 128
learning_rate = .001
concat = 1

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

for i in range(len(posx)):
    mol = cmol2(posx[i])
    mac = np.zeros(2048)
    DataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(posx[i], radius=2), mac)
    if isinstance(mol, int):
        print("dropped")
        print(i)
    else:
        dposx.append(mol)
        dposy.append(1)
        mposx.append(mac)
        mposy.append(1)
print(len(dposx))

for i in range(len(negx)):
    mol = cmol2(negx[i])
    mac = np.zeros(2048)
    DataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(negx[i], radius=2), mac)
    if isinstance(mol, int):
        print(i)
    else:
        dnegx.append(mol)
        dnegy.append(0)
        mnegx.append(mac)
        mnegy.append(0)

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

for i in range(len(mposx)):
    if (i < len(mposx)/5):
        postex.append(mposx[i])
        postey.append(1)
    elif (i < len(mposx)*.3):
        posvax.append(mposx[i])
        posvay.append(1)
    else:
        postrx.append(mposx[i])
        postry.append(1)
    print(i)
        
for i in range(len(mnegx)):
    if (i < len(mnegx)/5):
        negtex.append(mnegx[i])
        negtey.append(0)
    elif (i < len(mnegx)*.3):
        negvax.append(mnegx[i])
        negvay.append(0)
    else:
        negtrx.append(mnegx[i])
        negtry.append(0)
    print(i)


mX_train = np.array(negtrx+postrx*31)
mX_val = np.array(negvax+posvax*31)
mX_test = np.array(negtex+postex)

my_train_s = np.array(negtry+postry*31)
my_val_s = np.array(negvay+posvay*31)
my_test_s = np.array(negtey+postey)



mmodel = keras.models.Sequential()
mmodel.add(Dense(512, input_dim=2048, activation='relu'))
mmodel.add(Dense(512, activation='relu'))
mmodel.add(Dense(512, activation='relu'))
mmodel.add(Dense(1, activation='sigmoid'))
    #Concatenate for longer epochs
Xt = np.concatenate([mX_train]*concat, axis=0)
yt = np.concatenate([my_train_s]*concat, axis=0)

steps_per_epoch = 10000/batch_size

optimizer = Adam(lr=learning_rate)
mmodel.summary()
print(len(mX_train))
print(len(mX_val))
print(len(mX_test))
mmodel.compile(loss='mse',optimizer=optimizer)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.25,patience=3, min_lr=1e-6, verbose=1)

start = time.time()
history = mmodel.fit(x = Xt, y = yt,
                                  epochs=epochs,
                                  validation_data=(mX_val,my_val_s),
                                  callbacks=[reduce_lr])
stop = time.time()
time_elapsed = stop - start

print("########################")
my_predict = mmodel.predict(mX_test)
my_predict = my_predict.reshape(len(my_predict))
print(np.count_nonzero(my_predict))
print(sklearn.metrics.roc_auc_score(my_test_s, my_predict))
print(np.mean(np.absolute(np.subtract(my_test_s, my_predict))))