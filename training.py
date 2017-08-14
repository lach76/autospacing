"""
    BiDirectional RNN
"""
import time
import math
import tensorflow as tf
from tensorflow.contrib import rnn

class Training(object):
    class_0 = 0
    class_1 = 1
    class_size = 2

    iter_num = 50
    display_step = 100

    hidden_size1 = 128
    hidden_size2 = 16  # hidden layer num of features

    learning_rate = 0.01
    batch_size = 100
    sequence_size = 20
    vector_size = 100
    drop_out_rate = 0.7

    train_file = "rsc/train_fixed.txt"
    test_file = "rsc/test_fixed.txt"

    whitespace_model_file = "model/whitespace.ckpt"
    w2v_file = "model/w2v.cbow"

    line_count = 0
    char_len = 0
    sentence = ""
    label = ""

    def __init__(self, learning_rate, batch_size, sequence_size, vector_size, drop_out_rate = 0.7):
        self.learning_rate = 0.01
        self.batch_size = 100
        self.sequence_size = 20
        self.vector_size = 100
        self.drop_out_rate = 0.7

        self.hidden_size1 = 128
        self.hidden_size2 = 16

        self.class_0 = 0
        self.class_1 = 1
        self.class_size = 2

    def getNextBatch(self, index, batchSize, inFilename, w2vModel):
        count = 0
        sentences = []
        labels = []
        start = index * batchSize
        end = (index + 1) * batchSize

        f = open(inFilename, 'r')
        while True:
            line = f.readline().strip('\n')
            if not line: break

            if count < start:
                count += 1
                continue
            if count >= end: break

            sentence = ''
            label = ''
            if line.count('\t') == 1:
                sentence, label = line.split('\t')
            elif line.count('\t') == 2:
                sentence1, sentence2, label = line.split('\t')
                sentence = sentence1 + '\t' + sentence2

            umjuls = [w2vModel.wv[w] for w in sentence]
            sentences.append(umjuls)

            umjuls = [w for w in label]
            labels.append(umjuls)
            count += 1

        f.close()

        inputX = util.pad_Xsequences_test(sentences, self.sequence_size)
        inputY = util.pad_Ysequences_test(labels, self.sequence_size)

        return inputX, inputY

    def BiRNN(self, X):
        X = tf.unstack(X, self.sequence_size, axis=1)

        fw_cell = rnn.GRUCell(self.hidden_size1)
        bw_cell = rnn.GRUCell(self.hidden_size1)

        outputs, state_fw, state_bw = rnn.static_bidirectional_rnn(cell_fw=fw_cell, cell_bw=bw_cell, inputs=X, dtype=tf.float32)

        self.weights1 = tf.Variable(tf.truncated_normal([2 * self.hidden_size1, self.hidden_size2], stddev=math.sqrt(2.0 / (2 * self.hidden_size1))))
        self.weights2 = tf.Variable(tf.truncated_normal([self.hidden_size2, self.class_size], stddev=math.sqrt(2.0 / (2 * self.hidden_size1))))
        self.biases1 = tf.Variable(tf.random_uniform([self.hidden_size2], -1.0, 1.0), name="bias_out1")
        self.biases1 = tf.Variable(tf.random_uniform([self.class_size], -1.0, 1.0), name="bias_out2")

        final_outputs = []
        for output in outputs:
            hidden1   = tf.nn.relu( tf.matmul( output, self.weights1) + self.biases1 )
            hidden1_1 = tf.nn.dropout(hidden1, self.drop_out_rate)
            out   = tf.nn.relu( tf.matmul( hidden1_1, self.weights2) + self.biases2 )
            final_outputs.append(out)
        return final_outputs

    def model(self):
        x = tf.placeholder("float", [None, self.sequence_size, self.vector_size], name="x")
        y = tf.placeholder("int32", [None, self.sequence_size], name="y")

        y_ = self.BiRNN(x)
        logits = tf.reshape(tf.concat(y_, 1), [-1, self.class_size], name="logits")
        targets = tf.reshape(y, [1, -1], name="targets")

        seq_weights = tf.ones([self.batch_size * self.sequence_size])
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits], [targets], [seq_weights])

        cost = tf.reduce_sum(loss) / self.batch_size
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, name="optimizer").minimize(cost)


    def train(self, x_data, y_data):

        session = tf.InteractiveSession()
        #session = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        epoch = 1
        start_time = time.time()
        while epoch <= self.training_iters:
            for i in range(0, self.iter_size):
                batch_xs, batch_ys = self.getNextBatch(i, self.batch_size, self.train_file)
                feed = {self.x: batch_xs, self.y: batch_ys}
                session.run(self.optimizer, feed_dict=feed)

            if epoch % self.display_step == 0:
                end_time = time.time()
                loss_total = 0
                for index in range(0, self.iter_size):
                    batch_xs, batch_ys = self.getNextBatch(i, self.batch_size, self.train_file)
                    feed = {x: batch_xs, y: batch_ys}
                    loss_value = session.run([self.cost], feed_dict=feed)
                    loss_total += loss_value[0]

                print('epoch : %s' % epoch + ',' + ' cost : %s' % (loss_total/self.iter_size) + ', time: %0.2f' % (end_time - start_time))
                saver.save(session, self.whitespace_model_file)
                start_time = time.time()

            epoch += 1

        saver.save(session, self.whitespace_model_file)
        print ('save 8000(final)')



