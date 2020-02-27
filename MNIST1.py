#https://tensorflowkorea.gitbooks.io/tensorflow-kr/content/g3doc/tutorials/mnist/beginners/
#tensorflow 2.0버전 기준
#from tensorflow.examples.tutorials.mnist import input_data가 먹히지 않아
#따로 input_data 패키지를 추가한 후 진행했습니다.

import tensorflow.compat.v1 as tf
import input_data

tf.disable_v2_behavior()
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


x = tf.placeholder(tf.float32, [None, 784]) # 28x28 픽셀을 일렬로 나열, 차원의 갯수가 몇개인지는 정하지 않음.
#W,b는 학습시킬 데이터
W = tf.Variable(tf.zeros([784, 10])) #784차원을 10차원으로 변환하기 위함
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W)+b)

y_ = tf.placeholder(tf.float32, [None, 10]) #올바른 답이 들어갈 공간
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1])) #y의 각 원소의 로그 값 X (y_원소) + y의 두번째 차원
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for i in range(1000):
    # xs는 총 55000개의 이미지 -> 그래서 x를 None으로 선언
    # ys는 어떤 숫자인지 알려주는 라벨 -> 정확도 판단?
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

sess.close()