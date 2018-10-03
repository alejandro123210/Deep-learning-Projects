# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
import numpy
numpy.random.seed(7)

dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")

X = dataset[:,0:8]
Y = dataset[:,8]

#creating model 
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X,Y, epochs=150, batch_size=10, verbose=2)

predictions = model.predict(X)

rounded = [round(x[0]) for x in predictions]
print(rounded)

# scores = model.evaluate(X,Y)
# print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))



