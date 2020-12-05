import numpy as np
import pandas as pd
from nump import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from datascience import *

#load data
data = np.loadtxt('disprot_features.tsv', skiprows=1, usecols=list(range(2, 41)))
#name labels and features
labels = data[:, 0]
features = data[:, 1:]
#making table with features
features_names = np.loadtxt('disprot_features.tsv', dtype ='str',comments='#', delimiter=None, converters=None, skiprows=0, usecols=None, unpack=False, ndmin=0, encoding='bytes', max_rows=1)[3:]
features_names = np.delete(features_names, 6)
features_data = pd.DataFrame(features, columns = features_names)
features_data = Table.from_df(features_data)
#table with labels with features
features_with_labels = features_data.with_column('Disorder', labels)
#table with only disordered proteins
disordered_data = features_with_labels.where('Disorder', 1)
#table with only ordered protiens
ordered_data = features_with_labels.where('Disorder', 0)

#define model
model = Sequential()
model.add(Dense(12, input_dim=38, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#compile keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the keras model on the dataset
model.fit(features, labels, epochs=2, batch_size=10)

predictions = model.predict_classes(features)
for i in np.arange(10):
    print('%s => %d (expected %d)' % (features[i].tolist(), predictions[i], labels[i]))






