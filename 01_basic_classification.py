# tensorflow tutorials
# 1. basic classification
# https://www.tensorflow.org/tutorials/keras/basic_classification

from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)


############ data
print("\n--------------- data ---------------\n")
# load data
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# exploring data
print("train_images.shape: " + str(train_images.shape))
print("len(train_labels): " + str(len(train_labels)))
print("train_labels: " + str(train_labels))

print("test_images.shape: " + str(test_images.shape))
print("len(test_labels): " + str(len(test_labels)))
print("test_labels: " + str(test_labels))


############ preprocessing data
print("\n--------------- preprocessing data ---------------\n")
# look at first image
# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()

# scale from 0 to 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# look at first 25 images and their class names - verify they are correct
# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()


############ build the model
print("\n--------------- build model ---------------\n")
# setup the layers
# Flatten layer reformats the data from a 28x28 array to a 1x784 array
# Second dense layer is a fully connected neural layer with 128 nodes
# Final dense layer has 10 nodes (a softmax layer) which returns an array of 10 probabilities, summing to 1. Each nodes contains the probability score that the tested image belongs to a particular class.
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

# compile the model
# optimizer - how the model is updated based on the loss function
# loss function - measures how accurate the model is during training
# metrics - what are we measuring
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


############ train the model
print("\n--------------- train model ---------------\n")
# train the model on the training data
model.fit(train_images, train_labels, epochs=5)

############ evaluate accuracy
print("\n--------------- evaluate accuracy ---------------\n")
# test the model on the test data
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)

# test accuracy is less than training accuracy showing that we are overfitting

############ make predictions
print("\n--------------- make predictions ---------------\n")
predictions = model.predict(test_images)
print("predictions[0]: " + str(predictions[0]))
print("np.argmax(predictions[0]): " + str(np.argmax(predictions[0])))
print("test_labels[0]: " + str(test_labels[0]))

# finction for displaying the image
def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  
  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'
  
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

# function for looking at the 10 channels
def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1]) 
  predicted_label = np.argmax(predictions_array)
 
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

# looking at the 0th image
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)
plt.show()

# looking at the 12th image
i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)
plt.show()

# Plot the first X test images, their predicted label, and the true label
# Color correct predictions in blue, incorrect predictions in red
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)
plt.show()

# making a prediction for a single image
img = test_images[0]
print("img.shape: " + str(img.shape))

# Add the image to a batch where it's the only member (keras only works on batches)
img = (np.expand_dims(img,0))
print("img.shape: " + str(img.shape))

predictions_single = model.predict(img)
print("predictions_single: " + str(predictions_single))

plot_value_array(0, predictions_single, test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)

print("np.argmax(predictions_single[0]):" + str(np.argmax(predictions_single[0])))

