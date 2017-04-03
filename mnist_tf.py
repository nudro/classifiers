# -*- coding: utf-8 -*-
"""
Spyder Editor

In this tutorial, we're going to train a model to look at images and predict what digits they are. 

MNIST number recognition regression problem using softmax. 
predict y is a digit from 0 - 10. 
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

"""
Onehot encode to turn the digit features 1- 10 as dummy data variables

x isn't a specific value. It's a placeholder, a value that we'll i
nput when we ask TensorFlow to run a computation. We want to be able 
to input any number of MNIST images, each flattened into a 784-dimensional vector. 
We represent this as a 2-D tensor of floating-point numbers, with a shape [None, 784]. (
Here None means that a dimension can be of any length.)

Each image is 28 pixels by 28 pixels. We can interpret this as a big array of numbers:
We can flatten this array into a vector of 28x28 = 784 numbers.
784 dimensional vector space

"""
x = tf.placeholder(tf.float32, [None, 784])

"""
Need the weights and biases: A Variable is a modifiable tensor that lives in 
TensorFlow's graph of interacting operations. It can be used and even modified by t
he computation. For machine learning applications, one generally has the model 
parameters be Variables.

y = softmax(Wx + b)
yn = softmax[wn * xn + bn] (several n times over)

"""
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

"""
we initialize both W and b as tensors full of zeros. 
Since we are going to learn W and b, it doesn't matter very much what they initially are.
"""

#implement the model 
"""
First, we multiply x by W with the expression tf.matmul(x, W). 
This is flipped from when we multiplied them in our equation, where we had Wx, 
as a small trick to deal with x being a 2D tensor with multiple inputs. 
We then add b, and finally apply tf.nn.softmax.
"""
y = tf.nn.softmax(tf.matmul(x, W) + b)

#Train
"""
To measure model loss we use cross-entropy: 
    Hy'(y) = -Sigma (y'i) * log(yi)
    where y': True Value
    and y: Predicted Value (one of the one-hot-encoded digits)
To implement cross-entropy we need to first add a new placeholder to input the correct answers.
    
"""
y_ = tf.placeholder(tf.float32, [None, 10])

#Then we can implement the cross-entropy function
"""
First, tf.log computes the logarithm of each element of y. 
Next, we multiply each element of y_ with the corresponding element of tf.log(y). 
Then tf.reduce_sum adds the elements in the second dimension of y, due to the reduction_indices=[1] parameter. 
Finally, tf.reduce_mean computes the mean over all the examples in the batch.

See commented model loss formula above: 
    
"""
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

"""
Now that we know what we want our model to do, it's very easy to have TensorFlow train it to do so. 

Because TensorFlow knows the entire graph of your computations, it can automatically use 
the backpropagation algorithm to efficiently determine how your variables affect the loss 
you ask it to minimize. 

Then it can apply your choice of optimization algorithm to modify 
the variables and reduce the loss.

In this case, we ask TensorFlow to minimize cross_entropy using the gradient descent algorithm 
with a learning rate of 0.5. Gradient descent is a simple procedure, where TensorFlow simply 
shifts each variable a little bit in the direction that reduces the cost. But TensorFlow also 
provides many other optimization algorithms: using one is as simple as tweaking one line.

List of other optimizers: https://www.tensorflow.org/api_guides/python/train#optimizers
"""
#train with gradient descent optimizer

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

"""
We can now launch the model in an InteractiveSession:

"""
sess = tf.InteractiveSession()

#Let's train -- we'll run the training step 1000 times!
#We first have to create an operation to initialize the variables we created:

tf.global_variables_initializer().run()

"""
Each step of the loop, we get a "batch" of one hundred random data points from our training set. 
We run train_step feeding in the batches data to replace the placeholders. Using small batches of 
random data is called stochastic training -- in this case, stochastic gradient descent. 

"""
for _ in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

#Evaluation 
"""
Well, first let's figure out where we predicted the correct label. 

tf.argmax is an extremely useful function which gives you the index of the highest entry 
in a tensor along some axis. 
For example, tf.argmax(y,1) is the label our model thinks is most likely for each input, 
while tf.argmax(y_,1) is the correct label. We can use tf.equal to check if our prediction matches the truth.

"""

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

"""
That gives us a list of booleans. To determine what fraction are correct, 
we cast to floating point numbers and then take the mean. For example, 
[True, False, True, True] would become [1,0,1,1] which would become 0.75.

"""

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

"""
Finally, we ask for our accuracy on our test data.

"""
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

"""
92% accuracy which isn't very good, per Google.  List of other methods to use
for this MNIST problem, here: 
    http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html

"""


