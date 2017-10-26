# !usr/bin/env python
# -*- coding:utf-8 -*-
import tensorflow as tf
from numpy import *
from sklearn.metrics import confusion_matrix
from sklearn.metrics import *
from GCForest import *

train_data=loadtxt("F:/test/FlavorTagging/train_10k.csv",delimiter=",")
test_data=loadtxt("F:/test/FlavorTagging/test_10k.csv",delimiter=",")
train_X,test_X=train_data[:,3:],test_data[:,3:]
train_y,test_y=train_data[:,:3],test_data[:,:3]
#print(train_X.shape)

"""添加网络层"""
def add_layer(inputs, in_size, out_size, activation_function=None,lamb=0.001):
    Weights = tf.Variable(tf.random_normal([in_size, out_size],dtype=tf.float32))
    #tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(lamb)(Weights))
    biases = tf.Variable(tf.constant(0.1, shape=[out_size]))
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


X=tf.placeholder(tf.float32,shape=[None,63])
Y=tf.placeholder(tf.float32,shape=[None,3])
keep_prob = tf.placeholder(tf.float32)
batch_size=100

"""定义前向传播"""
# l1=add_layer(X,63,200,activation_function=tf.nn.relu)
# l2=add_layer(l1,200,300,activation_function=tf.nn.relu)
# l3=add_layer(l2,300,100,activation_function=tf.nn.tanh)
# l4=add_layer(l3,100,300,activation_function=tf.nn.relu)
# l5=add_layer(l4,300,200,activation_function=tf.nn.tanh)
# l6=add_layer(l5,200,20,activation_function=tf.nn.relu)
# prediction=add_layer(l6,20,3,activation_function=None)

L1=tf.layers.dense(inputs=X,units=100,activation=tf.nn.relu)
l2=tf.layers.dense(inputs=L1,units=300,activation=tf.nn.tanh)
l3=tf.layers.dense(inputs=l2,units=500,activation=tf.nn.relu)
l4=tf.layers.dense(inputs=l3,units=300,activation=tf.nn.relu)
l5=tf.layers.dense(inputs=l4,units=80,activation=tf.nn.tanh)
prediction=tf.layers.dense(inputs=l5,units=3,activation=None)

"""定义反向传播"""
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=Y))
# tf.add_to_collection("losses",loss1)
# loss=tf.add_n(tf.get_collection("losses"))
train_op=tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

"""准确率计算"""
correct_pred=tf.equal(tf.argmax(prediction,1),tf.argmax(Y,1))
accuracy=tf.reduce_mean(tf.cast(correct_pred,dtype=tf.float32))


init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for j in range(1):
        for i in range(1000):
            count=i*batch_size
            count1=count+100
            batch_X,batch_y=train_X[count:count1],train_y[count:count1]
            sess.run(train_op,feed_dict={X:batch_X,Y:batch_y,keep_prob:0.5})
            sess.run(loss, feed_dict={X: batch_X, Y: batch_y, keep_prob: 0.5})
            if i%100==0:
                print(j,sess.run(accuracy,feed_dict={X:test_X,Y:test_y,keep_prob:0.5}))

    # accuracy,prediction=(sess.run([accuracy,prediction],feed_dict={X:test_X,Y:test_y,keep_prob:0.5}))
    # print("accuracy:", accuracy)
    # prediction2=argmax(prediction, 1)
    # test_y2=argmax(test_y,1)
    # result = confusion_matrix(test_y2, prediction2)
    # TPR = 1.0 * result[0, 0] / (result[0, 0] + result[0, 1])
    # TNR = 1.0 * result[1, 1] / (result[1, 0] + result[1, 1])
    # auc = 0.5 * (TPR + TNR)
    # print("auc:", auc)

    """增加一个级联森林"""
    import numpy as np
    prediction2 = tf.nn.softmax(prediction)
    dense_train_X = sess.run(prediction2, feed_dict={X: train_X, Y: train_y, keep_prob: 0.7})
    dense_test_X = sess.run(prediction2, feed_dict={X: test_X, Y: test_y, keep_prob: 0.7})
    dense_feature_train = tf.reshape(dense_train_X, [-1, 3]).eval(session=sess)
    dense_feature_test = tf.reshape(dense_test_X, [-1, 3]).eval(session=sess)
    print(dense_feature_train.shape)
    print(dense_feature_test.shape)

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
    print(accuracy_score(y_true=test_y, y_pred=preds))



