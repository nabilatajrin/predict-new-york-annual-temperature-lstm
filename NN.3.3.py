import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import  Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras import metrics

df = pd.read_csv('NY.csv')
TAVG = np.array([df.iloc[:, 6]])
TMAX = np.array([df.iloc[:, 5]])
TMIN = np.array([df.iloc[:, 4]])
print(TAVG)

fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(TMAX, TMIN, TAVG, marker='o')
ax.set_xlabel('TMIN')
ax.set_ylabel('TMAX')
ax.set_zlabel('TAVG')
plt.savefig('fig1')
plt.plot(TAVG[0,:],TAVG[0,:]) #not working
plt.savefig('fig2')

X = np.concatenate([TMIN, TMAX], axis=0)
X = np.transpose(X)
Y = np.transpose(TAVG)
sc = MinMaxScaler()
sc.fit(X)
X = sc.transform(X)
sc1 = MinMaxScaler()
sc1.fit(Y)
Y = sc1.transform(Y)
X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

model = Sequential()
model.add(LSTM(20, activation='tanh', input_shape=(1, 2), recurrent_activation='hard_sigmoid'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=[metrics.mae])
model.fit(X_train, Y_train, epochs=50, verbose=2)

predict = model.predict(X_test)
plt.figure(2)
plt.scatter(Y_test, predict)
#plt.savefig(block=False)
plt.savefig('fig3')
plt.figure(3)
Real = plt.plot(Y_test)
Predict = plt.plot(predict)
plt.savefig('fig4')
