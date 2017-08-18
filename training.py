"""
    BiDirectional RNN

    8 hidden layers NN
        RELUs
        Number of nodes decrease by 50% with each hidden layer that is deeper in the neural net
    Overfitting measures
        L2 Regularization: Learning rate (beta) with exponential decay
        Dropout
    refer: http://www.ritchieng.com/machine-learning/deep-learning/tensorflow/regularization/
"""
import time
import math
import sys
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

class Training(object):
    class_0 = 0
    class_1 = 1
    class_size = 2

    hidden_size1 = 128
    hidden_size2 = 16  # hidden layer num of features

    learning_rate = 0.01
    batch_size = 2059
    iter_num = 5
    sequence_size = 20
    vector_size = 100
    drop_out_rate = 0.5
    display_step = 1
    training_iters = 10000
    early_stop_count = 3
    beta = 0.001

    model_file = ""
    w2v_file = ""

    line_count = 0
    char_len = 0
    sentence = ""
    label = ""

    def __init__(self, model_file, w2v_fle, learning_rate, batch_size, iter_num, sequence_size, vector_size, drop_out_rate = 0.7, early_stop_count=3):
        self.model_file = model_file
        self.w2v_fle = w2v_fle

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.iter_num = iter_num
        self.sequence_size = sequence_size
        self.vector_size = vector_size
        self.drop_out_rate = drop_out_rate
        self.early_stop_count = early_stop_count
        self.beta = 0.001

        self.hidden_size1 = 128
        self.hidden_size2 = int(self.hidden_size1 * np.power(0.5, 3))  # 16

        self.class_0 = 0
        self.class_1 = 1
        self.class_size = 2

        self.buildModel()
        self.saver = tf.train.Saver()

    def BiRNN(self, X):
        X = tf.unstack(X, self.sequence_size, axis=1)

        fw_cell = rnn.GRUCell(self.hidden_size1)
        bw_cell = rnn.GRUCell(self.hidden_size1)

        outputs, state_fw, state_bw = rnn.static_bidirectional_rnn(cell_fw=fw_cell, cell_bw=bw_cell, inputs=X, dtype=tf.float32)

        self.weights1 = tf.Variable(tf.truncated_normal([2 * self.hidden_size1, self.hidden_size2], stddev=math.sqrt(2.0 / (2 * self.hidden_size1))))
        self.weights2 = tf.Variable(tf.truncated_normal([self.hidden_size2, self.class_size], stddev=math.sqrt(2.0 / (2 * self.hidden_size2))))
        self.biases1 = tf.Variable(tf.random_uniform([self.hidden_size2], -1.0, 1.0), name="bias_out1")
        self.biases2 = tf.Variable(tf.random_uniform([self.class_size], -1.0, 1.0), name="bias_out2")

        final_outputs = []
        for output in outputs:
            hidden1   = tf.nn.relu( tf.matmul(output, self.weights1) + self.biases1)
            hidden1_1 = tf.nn.dropout(hidden1, self.drop_out_rate)
            out   = tf.nn.relu( tf.matmul(hidden1_1, self.weights2) + self.biases2)
            final_outputs.append(out)
        return final_outputs

    def buildModel(self):
        self.x = tf.placeholder("float", [None, self.sequence_size, self.vector_size], name="x")
        self.y = tf.placeholder("int32", [None, self.sequence_size], name="y")

        y_ = self.BiRNN(self.x)
        self.logits = tf.reshape(tf.concat(y_, 1), [-1, self.class_size], name="logits")
        targets = tf.reshape(self.y, [1, -1], name="targets")

        seq_weights = tf.ones([self.batch_size * self.sequence_size])
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([self.logits], [targets], [seq_weights])
        self.cost = tf.reduce_sum(loss)
        regularizers = tf.nn.l2_loss(self.weights1) + tf.nn.l2_loss(self.weights2)
        self.cost = tf.reduce_mean(self.cost + self.beta * regularizers) / self.batch_size

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, name="optimizer").minimize(self.cost)

    def getNextBatch(self, index, x, y):
        return x[index*self.batch_size:(index+1)*self.batch_size], y[index*self.batch_size:(index+1)*self.batch_size]

    def train(self, x_data, y_data):
        session = tf.InteractiveSession()
        session.run(tf.global_variables_initializer())

        early_stop_count = 0
        previous_loss = sys.float_info.max
        epoch = 1
        start_time = time.time()
        while epoch <= self.training_iters:
            for i in range(self.iter_num):
                batch_xs, batch_ys = self.getNextBatch(i, x_data, y_data)
                feed = {self.x: batch_xs, self.y: batch_ys}
                session.run(self.optimizer, feed_dict=feed)

            if epoch % self.display_step == 0:
                end_time = time.time()
                loss_total = 0
                for i in range(self.iter_num):
                    batch_xs, batch_ys = self.getNextBatch(i, x_data, y_data)
                    feed = {self.x: batch_xs, self.y: batch_ys}
                    loss_value = session.run([self.cost], feed_dict=feed)
                    loss_total += loss_value[0]
                avg_loss = loss_total/self.iter_num

                if (avg_loss < previous_loss):
                    previous_loss = avg_loss
                    self.saver.save(session, self.model_file)
                else:
                    early_stop_count += 1
                    if (early_stop_count > self.early_stop_count):
                        break

                print('epoch : %s' % epoch + ',' + ' cost : %s' % avg_loss + ', time: %0.2f' % (end_time - start_time) + ', stop_count: %2d' % (early_stop_count))
                if (avg_loss < 0.01):
                    self.saver.save(session, self.model_file)
                    break
                start_time = time.time()

            epoch += 1

    def predict(self, x_data, sentence):
        session = tf.InteractiveSession()
        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(session, self.model_file)

        feed = {self.x: x_data}
        predict = session.run([tf.arg_max(self.logits, 1)], feed_dict=feed)

        line = ""
        diff = len(predict[0]) - len(sentence)
        for i in range(diff, len(predict[0])):
            if predict[0][i] == 0:
                line += sentence[i-diff]
            else:
                line += sentence[i-diff]+" "
        return line
