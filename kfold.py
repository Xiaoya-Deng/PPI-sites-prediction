# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 12:56:20 2019

@author: dengxy
"""

# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import os
import re

from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score


import inference


#parameters
INPUT_NODE = 434
RESHAPE_NODE = INPUT_NODE / 2
OUTPUT_NODE = 2


learning_rate = 0.01
global_step = tf.Variable(0, trainable = False)
decaylearning_rate = tf.train.exponential_decay(learning_rate, global_step, 10, 0.9)
    
#weight and bias
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


#Convolution layer

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='SAME')


#Pooling layer
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')  

#input
x = tf.placeholder(tf.float32, [None, INPUT_NODE]) 
y = tf.placeholder(tf.float32, [None, OUTPUT_NODE]) 

x_input = tf.reshape(x, [-1, 2, 217, 1])

#conv1
W_conv1 = weight_variable([3, 3, 1, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.softplus(conv2d(x_input, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


#conv2
W_conv2 = weight_variable([3, 3, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.softplus(conv2d(h_pool1, W_conv2) + b_conv2)

h_pool2 = max_pool_2x2(h_conv2)


#conv3
W_conv3 = weight_variable([1, 3, 64, 128])
b_conv3 = bias_variable([128])

h_conv3 = tf.nn.softplus(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)


#----fully connected layer----#
W_fc1 = weight_variable([ 1 * 4 * 128, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool3, [-1, 1 * 4 * 128])
h_fc1 = tf.nn.softplus(tf.matmul(h_pool2_flat,W_fc1) + b_fc1)


keep_prob = tf.placeholder("float")
h_fc1_dropout = tf.nn.dropout(h_fc1, keep_prob)

#output
W_fc2 = weight_variable([1024,OUTPUT_NODE])
b_fc2 = bias_variable([OUTPUT_NODE])
y_conv = tf.matmul(h_fc1_dropout, W_fc2) + b_fc2
pred = tf.nn.softmax(y_conv)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_conv))
train_step = tf.train.AdamOptimizer(decaylearning_rate).minimize(cross_entropy)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(y_conv, 1)), tf.float32))

trainAUC = []
testAUC = []


# 5-fold
#high binding propensity samples
"""
train_data = np.empty([0,INPUT_NODE + 2], dtype = float)

for root,dirs,files in os.walk(r"path/"):
    for file in files:
        path = os.path.join(root,file)
        #print(path)
        file_flag = re.search('_0', path)
        
        if file_flag is not None:
            sample = inference.readDataSet(path)
            train_data = np.vstack([train_data, sample])
            del sample
"""

#random sample
train_data = inference.readDataSet("sample.txt")

train_data = shuffle(train_data)            
label = train_data[:, -1].reshape(-1, 1)

kf = KFold(n_splits = 5)
n = 0

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
a = 0

for train_index, test_index in kf.split(train_data):
    n += 1
    print('train_index', train_index, 'test_index', test_index)
    train_X, train_y = train_data[train_index], label[train_index]
    test_X, test_y = train_data[test_index], label[test_index]
    
    train_rX, train_rY = inference.train_data(train_X)
    test_rX, test_rY = inference.train_data(test_X)
    print(len(train_rX))
    print(len(test_rX))
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        total_cost = 0
        for i in range(6000):
            x_batch, y_batch = inference.random_batch(train_rX, train_rY, 64)
            x_batch = x_batch.astype(np.float32)
            y_batch = y_batch.astype(np.float32)
            _,cross = sess.run([train_step,cross_entropy], feed_dict={x: x_batch, y: y_batch, keep_prob:0.5})

            if i % 500 == 0:
                train_accuracy = accuracy.eval(feed_dict={x:x_batch, y: y_batch, keep_prob:1.})
                print( "step %d, training accuracy %g, cost %g"%(i, train_accuracy,cross))
                
              
            
        print ("test accuracy %g"%accuracy.eval(feed_dict={x: test_rX, y: test_rY, keep_prob:1.}))
            
        test_prediction_value = sess.run(y_conv,feed_dict={x: test_rX, keep_prob:1.})

        y_true = inference.one_hot_to_label(test_rY)
        fpr, tpr, thresholds = roc_curve(y_true, test_prediction_value[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.3f)' % (a, roc_auc))

    a += 1
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='black',
         label=r'Mean ROC (AUC = %0.3f $\pm$ %0.3f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
#plt.title('High propensity 5-fold cross validation ROC curve')
plt.legend(loc="lower right")
plt.savefig('pic.jpg', dpi = 1200)
plt.show()
plt.close()
