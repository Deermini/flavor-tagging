# !usr/bin/env python
# -*- coding:utf-8 -*-

import tensorflow as tf
from numpy import *

train_data=loadtxt("F:/test/FlavorTagging/train_10k.csv",delimiter=",")
test_data=loadtxt("F:/test/FlavorTagging/test_10k.csv",delimiter=",")

# a=zeros((train_data.shape[0],1))
# b=zeros((test_data.shape[0],1))
# train_data=hstack((train_data,a))
# test_data=hstack((test_data,b))

train_X,test_X=train_data[:,3:],test_data[:20000,3:]
train_y,test_y=train_data[:,:3],test_data[:20000,:3]

"""define placeholder for input to network"""
X=tf.placeholder(tf.float32,shape=[None,63])
Y=tf.placeholder(tf.float32,shape=[None,3])
keep_prob = tf.placeholder(tf.float32)
batch_size=100
LR=0.001

feature=tf.reshape(X,[-1,63,1])
#feature=tf.reshape(X,[-1,66,1])

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={X: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={X: v_xs, Y: v_ys, keep_prob: 1})
    return result

"""LSTM"""
layer_num=1
lstm_cell=tf.contrib.rnn.BasicLSTMCell(num_units=32,forget_bias=1.0)
#dropout_lstm=tf.nn.rnn_cell.DeviceWrapper(lstm_cell,keep_prob)
#init_s = lstm_cell.zero_state(batch_size=batch_size, dtype=tf.float32)
dropout_lstm=tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell,output_keep_prob=keep_prob)
mlstm_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * layer_num, state_is_tuple=True)
init_state = mlstm_cell.zero_state(batch_size, dtype=tf.float32)
outputs, state=tf.nn.dynamic_rnn(cell=mlstm_cell,
                                      inputs=feature,
                                      initial_state=None,
                                      dtype=tf.float32,
                                      time_major=False)

prediction = tf.layers.dense(inputs=outputs[:, -1, :], units=3)


"""loss and accuracy"""
loss = tf.losses.softmax_cross_entropy(onehot_labels=Y, logits=prediction)
train_op = tf.train.AdamOptimizer(1e-4).minimize(loss)

#accuracy = tf.metrics.accuracy(labels=tf.argmax(Y, axis=1), predictions=tf.argmax(prediction, axis=1),)[1]
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for j in range(50):
        for i in range(1000):
            count=i*batch_size
            count1=count+batch_size
            batch_X,batch_y=train_X[count:count1],train_y[count:count1]
            sess.run([train_op,loss],feed_dict={X:batch_X,Y:batch_y,keep_prob:0.7})
            # if i%100==0:
            #     print(sess.run(loss,feed_dict={X:batch_X,Y:batch_y,keep_prob:0.7}))
            if i%100==0:
                print(compute_accuracy(test_X, test_y))

    # print 10 predictions from test data
    test_output = sess.run(prediction, {X: test_X[:10]})
    pred_y = argmax(test_output, 1)
    print(pred_y, 'prediction number')
    print(argmax(test_y[:10], 1), 'real number')






