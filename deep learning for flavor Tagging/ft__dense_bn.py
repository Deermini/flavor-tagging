# !usr/bin/env python
# -*- coding:utf-8 -*-
import tensorflow as tf
from numpy import *
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
import numpy as np
from sklearn.metrics import *
from GCForest import *


train_data=loadtxt("F:/test/FlavorTagging/train_10k.csv",delimiter=",")
test_data=loadtxt("F:/test/FlavorTagging/test_10k.csv",delimiter=",")
# data=vstack((train_data,test_data))
# random.shuffle(data)
# train=data[:,3:]
# target=data[:,:3]
# train_X,test_X,train_y,test_y=train_test_split(train,target,test_size=0.25,random_state=11)

train_X,test_X=train_data[:,3:],test_data[:,3:]
train_y,test_y=train_data[:,:3],test_data[:,:3]

X=tf.placeholder(tf.float32,shape=[None,63])
Y=tf.placeholder(tf.float32,shape=[None,3])
keep_prob = tf.placeholder(tf.float32)
batch_size=100
N_HIDDEN = 6
EPOCH = 12
LR = 0.01
ACTIVATION = tf.nn.relu
B_INIT = tf.constant_initializer(0.1)
tf_is_train = tf.placeholder(tf.bool, None)
hidden_units=[100,300,300,200,100,60]

class NN(object):
    def __init__(self, batch_normalization=False):
        self.is_bn = batch_normalization

        self.w_init = tf.random_normal_initializer(0., .1)  # weights initialization
        self.pre_activation = [X]
        if self.is_bn:
            self.layer_input = [tf.layers.batch_normalization(X, training=tf_is_train)]  # for input data
        else:
            self.layer_input = [X]
        for i in range(N_HIDDEN):  # adding hidden layers
            self.layer_input.append(self.add_layer(self.layer_input[-1], 100, ac=ACTIVATION))
        self.out = tf.layers.dense(self.layer_input[-1], 3, kernel_initializer=self.w_init, bias_initializer=B_INIT)
        self.loss=loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.out,labels=Y))

        # !! IMPORTANT !! the moving_mean and moving_variance need to be updated,
        # pass the update_ops with control_dependencies to the train_op
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train = tf.train.AdamOptimizer(LR).minimize(self.loss)

    def add_layer(self, x, out_size, ac=None):
        x = tf.layers.dense(x, out_size, kernel_initializer=self.w_init, bias_initializer=B_INIT)
        self.pre_activation.append(x)
        # the momentum plays important rule. the default 0.99 is too high in this case!
        if self.is_bn: x = tf.layers.batch_normalization(x, momentum=0.4, training=tf_is_train)    # when have BN
        out = x if ac is None else ac(x)
        return out

nets = [NN(batch_normalization=False), NN(batch_normalization=True)]
prediction=nets[1].out

"""准确率计算"""
correct_pred=tf.equal(tf.argmax(prediction,1),tf.argmax(Y,1))
accuracy=tf.reduce_mean(tf.cast(correct_pred,dtype=tf.float32))

init=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for j in range(20):
        for i in range(1000):
            count=i*batch_size
            count1=count+batch_size
            batch_X,batch_y=train_X[count:count1],train_y[count:count1]
            sess.run(nets[1].train,feed_dict={X:batch_X,Y:batch_y,tf_is_train: True})
            sess.run(nets[1].loss, feed_dict={X: batch_X, Y: batch_y, tf_is_train: True})
            if i%100==0:
                print(j,sess.run(accuracy,feed_dict={X:test_X,Y:test_y,tf_is_train: False}))

    # accuracy,prediction=(sess.run([accuracy,prediction],feed_dict={X:test_X,Y:test_y,tf_is_train: False}))
    # print("accuracy:", accuracy)
    # prediction2=argmax(prediction, 1)
    # test_y2=argmax(test_y,1)
    # result = confusion_matrix(test_y2, prediction2)
    # TPR = 1.0 * result[0, 0] / (result[0, 0] + result[0, 1])
    # TNR = 1.0 * result[1, 1] / (result[1, 0] + result[1, 1])
    # auc = 0.5 * (TPR + TNR)
    # print("auc:", auc)

    """增加一个级联森林"""

    dense_train_X = sess.run(prediction, feed_dict={X: train_X, Y: train_y, tf_is_train: False})
    dense_test_X = sess.run(prediction, feed_dict={X: test_X, Y: test_y, tf_is_train: False})
    dense_train_X2= tf.nn.softmax(dense_train_X)
    dense_test_X2 = tf.nn.softmax(dense_test_X)
    dense_feature_train = tf.reshape(dense_train_X2, [-1, 3]).eval(session=sess)
    dense_feature_test = tf.reshape(dense_test_X2, [-1, 3]).eval(session=sess)

    """组合特征"""
    # print(LSTM_feature_train[1])
    new_train_X = np.hstack((train_X, dense_feature_train))
    new_test_X = np.hstack((test_X, dense_feature_test))
    """把label转换成0,1,2"""
    train_y = where(train_y == 1)[1]
    test_y = where(test_y == 1)[1]

    gcf = gcForest(tolerance=0.09, n_cascadeRFtree=101)
    _ = gcf.cascade_forest(new_train_X, train_y)
    pred_proba = gcf.cascade_forest(new_test_X)
    tmp = np.mean(pred_proba, axis=0)
    preds = np.argmax(tmp, axis=1)
    print("accurcy:",accuracy_score(y_true=test_y, y_pred=preds))

    result = confusion_matrix(test_y, preds)
    TPR = 1.0 * result[0, 0] / (result[0, 0] + result[0, 1])
    TNR = 1.0 * result[1, 1] / (result[1, 0] + result[1, 1])
    auc = 0.5 * (TPR + TNR)
    print("auc:", auc)



