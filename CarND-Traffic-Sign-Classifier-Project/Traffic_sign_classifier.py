import numpy as np

# Load pickled data
import pickle

# TODO: Fill this in based on where you saved the training and testing data

training_file = 'dataset/train.p'
validation_file = 'dataset/valid.p'
testing_file = 'dataset/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

assert(len(X_train) == len(y_train))
assert(len(X_valid) == len(y_valid))
assert(len(X_test) == len(y_test))

### Replace each question mark with the appropriate value. 
### Use python, pandas or numpy methods rather than hard coding the results

import tensorflow as tf

EPOCHS = 10
BATCH_SIZE = 128

# TODO: Number of training examples
n_train = len(X_train)

# TODO: Number of validation examples
n_valid = len(X_valid)

# TODO: Number of testing examples.
n_test = len(X_test)

# TODO: What's the shape of an traffic sign image?
image_shape = X_train[0].shape

# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(set(train['labels']))

print("Number of training examples =", n_train)
print("Number of validation examples =", n_valid)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

### Preprocess the data here. It is required to normalize the data. Other preprocessing steps could include 
### converting to grayscale, etc.
### Feel free to use as many code cells as needed.

from sklearn.utils import shuffle

# convert to grayscale; didn't improve performance
#X_train[...,0] = np.dot(X_train[...,:3], [0.299, 0.587, 0.114])
#X_valid[...,0] = np.dot(X_valid[...,:3], [0.299, 0.587, 0.114])
#X_test[...,0] = np.dot(X_test[...,:3], [0.299, 0.587, 0.114])
#X_train = np.delete(X_train, [1, 2], 3)
#X_valid = np.delete(X_valid, [1, 2], 3)
#X_test = np.delete(X_test, [1, 2], 3)

X_train, y_train = shuffle(X_train, y_train)

# normalize
X_train = (X_train - np.mean(X_train))/np.std(X_train)
X_valid = (X_valid - np.mean(X_valid))/np.std(X_valid)
X_test = (X_test - np.mean(X_test))/np.std(X_test)

### Define your architecture here.
### Feel free to use as many code cells as needed.
from tensorflow.contrib.layers import flatten

def LeNet(x):
    C = int(x.shape[3])
    conv1 = tf.layers.conv2d(inputs=x, filters=6*C, kernel_size= 5, padding="valid", activation=tf.nn.relu)
    print("conv1 is ")
    print(conv1.shape)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=2, strides=2)
    print("pool1 is ")
    print(pool1.shape)
    conv2 = tf.layers.conv2d(inputs=pool1, filters=16*C, kernel_size=5, padding="valid", activation=tf.nn.relu)
    print("conv2 is ")
    print(conv2.shape)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=2, strides=2)
    print("pool2 is ")
    print(pool2.shape)
    flat2 = flatten(pool2)
    print("flat1 is ")
    print(flat2.shape)
    dense3 = tf.layers.dense(flat2, 120*C, activation=tf.nn.relu, use_bias=True)
    print("dense3 is ")
    print(dense3.shape)
    dense4 = tf.layers.dense(dense3, 84*C, activation=tf.nn.relu, use_bias=True)
    print("dense4 is ")
    print(dense4.shape)
    logits = tf.layers.dense(dense4, n_classes, use_bias=True)
    
    return logits

x = tf.placeholder(tf.float32, (None, X_train.shape[1], X_train.shape[2], X_train.shape[3]))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, n_classes)

### Train your model here.
### Calculate and report the accuracy on the training and validation set.
### Once a final model architecture is selected, 
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.
rate = 0.001

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

### Train your model here.
### Calculate and report the accuracy on the training and validation set.
### Once a final model architecture is selected, 
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.
rate = 0.001

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
            
        validation_accuracy = evaluate(X_valid, y_valid)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        
    saver.save(sess, './traffic')
    print("Model saved")
    
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('.'))

    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))

