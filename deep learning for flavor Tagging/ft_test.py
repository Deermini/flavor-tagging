# !usr/bin/env python
# -*- coding:utf-8 -*-
import tensorflow as tf
from numpy import *


train_data=loadtxt("F:/test/FlavorTagging/train_10k.csv",delimiter=",")
test_data=loadtxt("F:/test/FlavorTagging/test_10k.csv",delimiter=",")
train_X,test_X=train_data[:,3:],test_data[:,3:]
train_y,test_y=train_data[:,:3],test_data[:,:3]

def get_weight(shape,lamb):
    weight1=tf.Variable(tf.random_normal(shape),dtype=tf.float32)
    tf.add_to_collection("losses",tf.contrib.layers.l2_regularizer(lamb)(weight1))
    return weight1

X=tf.placeholder(tf.float32,shape=[None,63])
Y=tf.placeholder(tf.float32,shape=[None,3])

batch_size=100
layers_dimension=[63,100,3]
n_layers=len(layers_dimension)
cur_result=X
for i in range(n_layers-1):
    weights=get_weight([layers_dimension[i],layers_dimension[i+1]],0.001)
    biases=tf.Variable(tf.constant(0.1,shape=[layers_dimension[i+1]]))
    if i<n_layers-2:
        cur_result=tf.nn.sigmoid(tf.matmul(cur_result,weights)+biases)
    else:
        cur_result = tf.matmul(cur_result, weights) + biases
        pred=tf.nn.softmax(cur_result)

mse_loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=cur_result,labels=Y))
tf.add_to_collection("losses",mse_loss)
loss=tf.add_n(tf.get_collection("losses"))
train_step=tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

correct_pred=tf.equal(tf.argmax(cur_result,1),tf.argmax(Y,1))
accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))

init=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for j in range(60):
        for i in range(1000):
            count=i*batch_size
            count1=count+100
            batch_X,batch_y=train_X[count:count1],train_y[count:count1]
            sess.run(train_step,feed_dict={X:batch_X,Y:batch_y})
            if i%100==0:
                print(sess.run(loss,feed_dict={X:batch_X,Y:batch_y}))
    print("accuracy:",sess.run(accuracy,feed_dict={X:test_X,Y:test_y}))






