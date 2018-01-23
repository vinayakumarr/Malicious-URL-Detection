from __future__ import print_function
from sklearn.cross_validation import train_test_split
import pandas as pd
import numpy as np
np.random.seed(1337)  
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM, SimpleRNN, GRU
from keras.datasets import imdb
from keras.utils.np_utils import to_categorical
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score,mean_squared_error,mean_absolute_error)
from sklearn import metrics
from sklearn.preprocessing import Normalizer
import h5py
from keras import callbacks
from keras.callbacks import CSVLogger
import keras
import keras.preprocessing.text
import itertools
from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from keras import callbacks
from sklearn import feature_extraction
import sklearn

trainlabels = pd.read_csv('dgcorrect/trainlabel.csv', header=None)
trainlabel = trainlabels.iloc[:,0:1]
testlabels = pd.read_csv('dgcorrect/testlabel.csv', header=None)
testlabel = testlabels.iloc[:,0:1]

train = pd.read_csv('dgcorrect/train.txt', header=None)
test = pd.read_csv('dgcorrect/test.txt', header=None)

X = train.values.tolist()
X = list(itertools.chain(*X))

T = test.values.tolist()
T = list(itertools.chain(*T))

print("vectorizing data")
ngram_vectorizer = feature_extraction.text.CountVectorizer(analyzer='char', ngram_range=(2, 2))
X_train = ngram_vectorizer.fit_transform(X)
y_train = np.array(trainlabel)

print("vectorizing data")
ngram_vectorizer1 = feature_extraction.text.CountVectorizer(analyzer='char', ngram_range=(2, 2))
X_test = ngram_vectorizer1.fit_transform(T)
y_test = np.array(testlabel)

max_features = X_train.shape[1]


model = Sequential()
model.add(Dense(1, input_dim=max_features, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])

checkpointer = callbacks.ModelCheckpoint(filepath="logs/bigram/checkpoint-{epoch:02d}.hdf5", verbose=1, save_best_only=True, monitor='val_acc',mode='max')
csv_logger = CSVLogger('logs/bigram/training_set_lstmanalysis.csv',separator=',', append=False)
model.fit(X_train.todense(), y_train, batch_size=32, nb_epoch=1000,validation_split=0.33, shuffle=True,callbacks=[checkpointer,csv_logger])
model.save("logs/bigram/coomplemodel.hdf5")
t_probs = model.predict_proba(X_test.todense())
print(t_probs.shape)
t_auc = sklearn.metrics.roc_auc_score(y_test, t_probs)
print(t_auc)


y_pred = model.predict_classes(X_test.todense())
np.savetxt('bigrampredicted.txt', y_pred, fmt='%01d')
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred , average="binary")
precision = precision_score(y_test, y_pred , average="binary")
f1 = f1_score(y_test, y_pred, average="binary")

print("----------------------------------------------")
print("accuracy")
print("%.3f" %accuracy)
print("racall")
print("%.3f" %recall)
print("precision")
print("%.3f" %precision)
print("f1score")
print("%.3f" %f1)


