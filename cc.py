# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 18:46:28 2020

@author: Admin
"""
import pandas as pd
from sklearn.model_selection import train_test_split as tts
import numpy as np
import random
# from sklearn.svm import SVC
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

df = pd.read_csv('xtrain.csv')

data = []
for j in range(len(df.columns)):
	z = df[df.columns[j]].values
	y = z[-1]
	z = z[:-1]

	for i in range(len(z)//25):
		data.append([z[25*i:25*i+25],y])

random.shuffle(data)      

X = []
Y = []
for x,y in data:
	X.append(x)
	Y.append(y)

xtr,xte,ytr,yte = tts(X,Y,test_size = 0.2)


model = Sequential([Dense(64,activation='relu'),
					Dense(32,activation='relu'),
					Dense(1)
					])

model.compile(loss='mse',optimizer='adam',metrics=['mse','mae'])

history = model.fit(np.array(xtr),np.array(ytr),epochs=50)
pred = model.predict(np.array(xte))

df1 = pd.DataFrame()

pred_questions = []
for i in df.columns[:10]:
  pred_questions.append(df[i].iloc[:25])
pred_questions = np.array(pred_questions)

pred = model.predict(pred_questions)

for idx,i in enumerate(df.columns[:10]):
  df1[i] = pred[idx]
  
df1.to_csv('predicted_answers2.csv')

df = pd.read_csv('xtest.csv')