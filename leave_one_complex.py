# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import os
import re

from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
#from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
#from sklearn.model_selection import train_test_split


import inference


#parameters
INPUT_NODE = 434
RESHAPE_NODE = INPUT_NODE / 2
OUTPUT_NODE = 2




learning_rate = 0.01
global_step = tf.Variable(0, trainable = False)
decaylearning_rate = tf.train.exponential_decay(learning_rate, global_step, 10, 0.9)


def evaluation(prediction_value, y_true, pro, num, name):
    y_true = inference.one_hot_to_label(y_true)   
    fpr, tpr, thresholds_keras = roc_curve(y_true, prediction_value[:, 1])
    roc_auc = auc(fpr, tpr)
    print("AUC : ", roc_auc)
	plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, lw=1, label='ROC(area = %0.5f)' % (roc_auc)) 
    
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title(pro + ' ROC curve')
	plt.show()
    plt.close()

    return roc_auc
    
    
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
positive_index = []
protein = ["1ACB","1AK4","1ATN","1AVX","1AY7","1B6C","1BKD","1BUH","1BVN","1CGI","1CLV","1D6R","1DFJ","1E6E","1E96","1EAW","1EFN","1EWY","1F34","1F6M","1FC2","1FFW","1FLE","1FQ1","1FQJ","1GCQ","1GHQ","1GL1","1GLA","1GPW","1GRN","1GXD","1H1V","1H9D","1HE1","1HE8","1I2M","1IBR","1IRA","1J2J","1JIW","1JK9","1JTD","1JTG","1KAC","1KTZ","1KXQ","1LFD","1M10","1MAH","1MQ8","1NW9","1OC0","1OPH","1PPE","1PVH","1PXV","1QA9","1R0R","1R6Q","1R8S","1RKE","1S1Q","1SBB","1SYX","1T6B","1TMQ","1UDI","1US7","1WQ1","1XD3","1XQS","1YVB","1Z0K","1Z5Y","1ZHH","1ZHI","2A1A","2A5T","2A9K","2ABZ","2AJF","2AYO","2B42","2BTF","2C0L","2CFH","2FJU","2G77","2GAF","2GTP","2H7V","2HLE","2HQS","2HRK","2I25","2I9B","2IDO","2J0T","2J7P","2NZ8","2O3B","2O8V","2OOB","2OT3","2OUL","2OZA","2PCC","2SIC","2SNI","2UUY","2VDB","2X9A","2YVJ","2Z0E","3A4S","3BIW","3BX7","3CPH","3D5S","3DAW","3F1P","3FN1","3H2V","3K75","3PC8","3S9D","3SGQ","3VLB","4CPA","4FZA","4IZ7","4M76","7CEI", "1KXP", "1Y64", "1ZM4", "4H03"]

for pro in protein:   
    
    print("Leave one complex validatiob：", pro)
    #sample
    train_data = np.empty([0,INPUT_NODE + 2], dtype = float)
    test_x = np.empty([0,INPUT_NODE + 2], dtype = float)
        
    for root,dirs,files in os.walk(r"/path_data/"):
        for file in files:
            path = os.path.join(root,file)
            
            pro_flag = re.search(pro, path)
            
            if pro_flag is not None:
                test_positive = '/path_positive/'
                t_posi = inference.readDataSet(test_positive)
                test_negative = '/path_negative/'
                t_nega = inference.readDataSet(test_negative)
            elif pro_flag is None:    
                sample = inference.readDataSet(path)
                train_data = np.vstack([train_data, sample])
                del sample 

    test_x = np.vstack([t_posi, t_nega])
    del t_posi
    del t_nega
    
    train_data = shuffle(train_data)            
    label = train_data[:, -1].reshape(-1, 1)
           
    train_rX, train_rY = inference.train_data(train_data)
    test_rX, test_rY = inference.train_data(test_x)
    del test_x
            
    #session

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
				
		
		test_prediction_value = np.empty([0, 2], dtype = float)
		n = int(len(test_rX) / 2)
		
		t1 = sess.run(pred,feed_dict={x: test_rX[0: n], keep_prob:1.})
		t2 = sess.run(pred,feed_dict={x: test_rX[n: ], keep_prob:1.})
		test_prediction_value = np.vstack((t1, t2))
		
		test_auc = evaluation(test_prediction_value, test_rY, pro, num, "test")
