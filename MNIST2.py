#tensorflow 2.0버전 기준
#from tensorflow.examples.tutorials.mnist import input_data가 먹히지 않아
#따로 input_data 패키지를 추가한 후 진행했습니다.

import tensorflow.compat.v1 as tf

import input_data

tf.disable_v2_behavior()
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


x = tf.placeholder(tf.float32, shape=[None, 784]) # 28x28 픽셀 이미지 데이터
y_ = tf.placeholder(tf.float32, shape=[None, 10]) # 10차원 one-hot 벡터

W = tf.Variable(tf.zeros([784, 10])) #784 x 10 행렬
b = tf.Varibale(tf.zeros([10])) # 10차원 행렬

sess=tf.Session()
sess.run(tf.global_variables_initializer())
y = tf.nn.softmax(tf.matmul(x, W)+b) #입력 데이터 갯수 x 10의 행렬이 만들어질 것이다.(2차원 행렬의 곱)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y, reduction_indices=[1])))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

for i in range(1000):
    batch = mnist.train.next_batch(50)
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))