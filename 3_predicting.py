
import gensim
import tensorflow as tf
from tensorflow.contrib import rnn

import util

# Parameters
learning_rate = 0.0001
training_iters = 100
display_step = 10

CLASS_0 = 0
CLASS_1 = 1
class_size = 2

# Network Parameters
train_size = 8000
test_size = 2000

iter_size = 10
batch_size = 200


# data Parameters
#drop_out_rate = 0.7
sequence_size = 30  # timesteps
vector_size = 100  # vocaburary size
hidden_size1 = 128  # hidden layer num of features
hidden_size2 = 16  # hidden layer num of features


#train_file = "rsc/train_variable.txt"
#test_file = "rsc/test_variable.txt"
#train_file = "rsc/train_fixed.txt"
test_file = "rsc/test_variable.txt"
#test_file = "rsc/train_fixed.txt"
#test_file = "rsc/test_fixed.txt"
test_out_file = "rsc/test_variable.out"

whitespace_model_file = "./8000/whitespace.ckpt"
w2v_file = "8000/w2v.txt"

line_count = 0
char_len = 0
sentence = ""
label = ""


def getNextBatch(productName, sequenceSize, w2vModel):
    sentences = []

    umjuls = [w2vModel.wv[w] for w in productName]
    sentences.append(umjuls)

    inputX = util.pad_Xsequences_test(sentences, sequenceSize)

    return inputX

def BiRNN(X, weights, biases):
    X = tf.unstack(X, sequence_size, axis=1)

    fw_cell = rnn.GRUCell(hidden_size1)
    bw_cell = rnn.GRUCell(hidden_size1)

    outputs, state_fw, state_bw = rnn.static_bidirectional_rnn(fw_cell, bw_cell, X, dtype=tf.float32)

    final_outputs = []
    for output in outputs:
        hidden1   = tf.nn.relu( tf.matmul( output, weights['out1']) + biases['out1'] )
        #hidden1_1 = tf.nn.dropout(hidden1, drop_out_rate)
        out   = tf.nn.relu( tf.matmul( hidden1, weights['out2']) + biases['out2'] )
        final_outputs.append(out)
    return final_outputs


x = tf.placeholder("float", [None, sequence_size, vector_size], name="x")
y = tf.placeholder("int32", [None, sequence_size], name="y")

weights = {
    'out1': tf.get_variable("out1", shape=[2 * hidden_size1, hidden_size2], initializer=util.xavier_init(2 * hidden_size1, hidden_size2)),
    'out2': tf.get_variable("out2", shape=[hidden_size2, class_size], initializer=util.xavier_init(hidden_size2, class_size)),
    #'out1': tf.get_variable("out1", shape=[2 * hidden_size1, hidden_size2], initializer=tf.contrib.layers.xavier_initializer()),
    #'out2': tf.get_variable("out2", shape=[hidden_size2, class_size], initializer=tf.contrib.layers.xavier_initializer()),
}
biases = {
    'out1': tf.Variable(tf.random_uniform([hidden_size2], -1.0, 1.0), name="bias_out1"),
    'out2': tf.Variable(tf.random_uniform([class_size], -1.0, 1.0), name="bias_out2"),
    #'out1': tf.Variable(tf.constant(0.1, shape=[hidden_size2]), name="bias_out1"),
    #'out2': tf.Variable(tf.constant(0.1, shape=[class_size]), name="bias_out2"),
}

y_ = BiRNN(x, weights, biases)
logits = tf.reshape(tf.concat(y_, 1), [-1, class_size], name="logits")
targets = tf.reshape(y, [1, -1], name="targets")

seq_weights = tf.ones([batch_size * sequence_size])
loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits], [targets], [seq_weights])

cost = tf.reduce_sum(loss) / batch_size
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name="optimizer").minimize(cost)


# load w2v
w2v = gensim.models.Word2Vec.load(w2v_file)
print('load w2v file')

session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(session, whitespace_model_file)

outfp = open(test_out_file, 'w')
infp = open(test_file, 'r')
sentence_correct_count = 0
sentence_wrong_count = 0
sentence_count = 0

correct_count = 0
wrong_count = 0

while True:
    line = infp.readline().strip('\n')
    if not line: break

    sentence, label = line.split('\t')

    sentences = []
    tmp_sentences = []
    labels = []
    tmp_labels = []
    index = 0
    for i in range(0, len(sentence)):
        labels.append(label[i])
        if index < sequence_size:
            tmp_sentences.append(w2v.wv[sentence[i]])
            index += 1
        else:
            sentences.append(tmp_sentences)
            tmp_sentences = []
            index = 0
    if index > 0:
        sentences.append(tmp_sentences)
        labels.append(tmp_labels)

    inputX = util.pad_Xsequences_test(sentences, sequence_size)
    feed = {x: inputX}
    answer = session.run([tf.arg_max(logits, 1)], feed_dict=feed)

    labelSize = len(labels)
    end = len(answer[0])
    start = end - labelSize + 1
    index = 0
    isSame = True
    line = ''
    for i in range(start, end-1):
        if (int(labels[index]) == answer[0][i]):
            correct_count += 1
        else:
            isSame = False
            wrong_count += 1

        line += sentence[index]
        if (answer[0][i] == 1):
            line += ' '
        index += 1

    line += sentence[len(sentence)-1]
    if isSame:
        outfp.write('O')
        outfp.write('\t')
        outfp.write(line)
        outfp.write('\n')
        sentence_correct_count += 1
    else:
        outfp.write('X')
        outfp.write('\t')
        outfp.write(line)
        outfp.write('\n')
        sentence_wrong_count += 1

    if sentence_count % 100 == 0:
        print("count: ", sentence_count)
        print("   char. correct count: ", correct_count, ", char. wrong count: ", wrong_count)
        print("   char. precision: ", correct_count / (correct_count + wrong_count))
        print("   sentence correct count: ", sentence_correct_count, ", sentence wrong count: ", sentence_wrong_count)
        print("   sentence precision: ", sentence_correct_count / (sentence_correct_count + sentence_wrong_count))

    sentence_count += 1

infp.close()
outfp.close()
print()
print("count: ", sentence_count)
print("   char. correct count: ", correct_count, ", char. wrong count: ", wrong_count)
print("   char. precision: ", correct_count / (correct_count + wrong_count))
print("   sentence correct count: ", sentence_correct_count, ", sentence wrong count: ", sentence_wrong_count)
print("   sentence precision: ", sentence_correct_count / (sentence_correct_count + sentence_wrong_count))

session.close()


"""
[CBOW]
count:  200
   char. correct count:  3324 , char. wrong count:  356
   char. precision:  0.9032608695652173
   sentence correct count:  44 , sentence wrong count:  156
   sentence precision:  0.22

[SKIP GRAM]
count:  200
   char. correct count:  2977 , char. wrong count:  665
   char. precision:  0.8174080175727623
   sentence correct count:  5 , sentence wrong count:  195
   sentence precision:  0.025
"""