#!usr/bin/env python
# -*- coding:utf-8 -*-

from GCForest import *
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from numpy import *

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
#prediction2 = tf.layers.dense(inputs=outputs[:, -1, :], units=3,activation=tf.nn.softmax)

"""loss and accuracy"""
loss = tf.losses.softmax_cross_entropy(onehot_labels=Y, logits=prediction)
train_op = tf.train.AdamOptimizer(1e-4).minimize(loss)

#accuracy = tf.metrics.accuracy(labels=tf.argmax(Y, axis=1), predictions=tf.argmax(prediction, axis=1),)[1]
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for j in range(10):
        for i in range(1000):
            count=i*batch_size
            count1=count+batch_size
            batch_X,batch_y=train_X[count:count1],train_y[count:count1]
            sess.run([train_op,loss],feed_dict={X:batch_X,Y:batch_y,keep_prob:0.7})
            # if i%100==0:
            #     print(sess.run(loss,feed_dict={X:batch_X,Y:batch_y,keep_prob:0.7}))
            if i%100==0:
                print(compute_accuracy(test_X, test_y))
    prediction=tf.nn.softmax(prediction)
    LSTM_train_X=sess.run(prediction,feed_dict={X:train_X,Y:train_y,keep_prob:0.7})
    LSTM_test_X = sess.run(prediction, feed_dict={X: test_X, Y: test_y, keep_prob: 0.7})
    LSTM_feature_train=tf.reshape(LSTM_train_X,[-1,3]).eval(session=sess)
    LSTM_feature_test = tf.reshape(LSTM_test_X, [-1, 3]).eval(session=sess)

    """组合特征"""
    #print(LSTM_feature_train[1])
    new_train_X=np.hstack((train_X,LSTM_feature_train))
    new_test_X=np.hstack((test_X,LSTM_feature_test))
    """把label转换成0,1,2"""
    train_y=where(train_y==1)[1]
    test_y=where(test_y==1)[1]

    gcf = gcForest(tolerance=0.09,n_cascadeRFtree=101)
    _ = gcf.cascade_forest(new_train_X,train_y)
    pred_proba = gcf.cascade_forest(new_test_X)
    tmp = np.mean(pred_proba, axis=0)
    preds = np.argmax(tmp, axis=1)
    print(accuracy_score(y_true=test_y, y_pred=preds))





