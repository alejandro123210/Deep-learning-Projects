# TensorFlow and tf.keras
# imports everything needed
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
import numpy
# sets random seed to 7 for reproducing the same results every time
numpy.random.seed(7)

# loads the dataset for the pima indians, found in ./data
dataset = numpy.loadtxt("./data/pima-indians-diabetes.csv", delimiter=",")

# slices data:
# the first : means all rows, the 0:8 means columns 0-8, which means 8th column gets ignored
X = dataset[:,0:8]
# the first : means all rows, the 8 means ONLY the 8th column, in other words, the output. 
Y = dataset[:,8]

# creates model layer by layer
# model type, Sequential
model = Sequential()
# adds the first layer (Dense means that the layers are fully connected, every node connects to every node)
# The 12 means 12 neurons, input dim means 8 inputs (one for each part of the data) and activation is recitifier, meaning 
# that the layer will generalized based on a straight line?  
model.add(Dense(12, input_dim=8, activation='relu'))
# this adds a second Dense layer, with 8 neurons, and the same recitifier activation
model.add(Dense(8, activation='relu'))
# this is the final layer, so only 1 neuron, because there is a binary answer if someone has diabetes
# the activation for this layer is sigmoid, this is a function that only outputs an answer between 0 and 1, making it a good 
# activation function for specifically predictions, considering something can't have a 110% chance of happening. 
model.add(Dense(1, activation='sigmoid'))

# This sets up the model to be run efficiently on a computer depending on hardware, so this is the part that optimizes 
# using Tensorflow. 
# It's important to define the kind of loss used for optimal predictions, in this case,
# the loss in this model is lograithmic, defined as crossentropy
# Adam will be used as the gradient descent algorithm primarily because it's efficient 
# Finally, because this problem is classification, accuracy is the best metric to measure. 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# This fits the data to the model in order for the model to be trained, 
# epochs is the amount of iterations through the dataset while 
# batch size is the number of datapoints looked at before the weights are changed
# finally, verbose is just the progress bar. 
model.fit(X,Y, epochs=15, batch_size=10, verbose=2)


# scores is equal to the evaluation of the models predictions (Y) from the data (X)
scores = model.evaluate(X,Y)
# this prints what's shown in the console, in other words, the accuracy
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))



# # predictions is the model's predictions
# predictions = model.predict(X)

# # rounded is equal to the rounded version of predictions since it used the sigmoid function, 
# # rounded is always either 0 or 1
# rounded = [round(x[0]) for x in predictions]
# # this prints the predictions
# print(rounded)