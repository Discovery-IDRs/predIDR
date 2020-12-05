# first neural network with keras make predictions

from numpy import loadtxt
import pandas as pd
import numpy as np
from nump import loadtxt
from keras.models import Sequential
from keras.layers import Dense
# load the dataset
data = np.loadtxt('disprot_features.tsv', skiprows=1, usecols=list(range(2, 41)))
labels = data[:, 0]
features = data[:, 1:]

#dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')
# split into input (X) and output (y) variables
X = dataset[:,0:8]
y = dataset[:,8]
# define the keras model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(X, y, epochs=15, batch_size=10, verbose=0)
# make class predictions with the model
predictions = model.predict_classes(X)
# summarize the first 5 cases
for i in range(5):
	print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))