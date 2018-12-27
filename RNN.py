import numpy as np
import pandas as pd

milk = pd.read_csv('monthly-milk-production.csv', index_col='Month')
# convert to time series data
milk.index = pd.to_datetime(milk.index)
train_data = milk[:-12]
test_data = milk.tail(12)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
train_set = scaler.fit_transform(train_data)
test_set = scaler.transform(test_data)

def next_batch(training_data, steps):
    start = np.random.randint(0, len(training_data)-steps)
    data_batch = np.array(training_data[start:start+steps+1]).reshape(1,steps+1)
    return data_batch[:,:-1].reshape(-1,steps,1), data_batch[:,1:].reshape(-1,steps,1)

import tensorflow as tf
n_in = 1
n_steps = 12
n_neuro = 100
n_out = 1
learning_rate = 0.001
iter = 6000
batch_size = 1
# steps - 1 to predict one step in the future
x = tf.placeholder(tf.float32, [None,n_steps,n_in])
y = tf.placeholder(tf.float32, [None, n_steps, n_out])
layer = tf.contrib.rnn.OutputProjectionWrapper(
    tf.contrib.rnn.BasicLSTMCell(num_units=n_neuro, activation=tf.nn.relu),
    output_size=n_out)
output, state = tf.nn.dynamic_rnn(layer, x, dtype=tf.float32)
loss = tf.reduce_mean(tf.square(output-y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(loss)
init = tf.global_variables_initializer()
saver = tf.train.Saver()

# with tf.Session() as sess:
#     sess.run(init)
#     for i in range(iter):
#         x_data, y_data = next_batch(train_set, n_steps)
#         sess.run(train, feed_dict={x:x_data, y:y_data})
#         if i % 100 == 0:
#             mse = loss.eval(feed_dict={x:x_data, y:y_data})
#             print(mse)
#     saver.save(sess, './RNN_model')

with tf.Session() as sess:
    saver.restore(sess, './RNN_model')
    train_seed = list(train_set[-12:])
    for i in range(12):
        x_data = np.array(train_seed[-12:]).reshape(1,n_steps,1)
        preds = sess.run(output, feed_dict={x:x_data})
        train_seed.append(preds[0,-1,0])

result = scaler.inverse_transform(np.array(train_seed[12:]).reshape(12,1))
test_data['predictions'] = result
test_data.plot()