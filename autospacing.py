import os

import gensim

from Autospacing.training import Training
from Util.preprocessing import Preprocessor

TRAINING_DATE_RATE = 1.0
SEQUENCE_SIZE = 50
VECTOR_SIZE = 100  # vocaburary size
RAW_DATA_FILE = "../resources/poems.txt"
TRAIN_FIXED_FILE = "../resources/autospacing_train_fixed.txt"
TEST_FIXED_FILE = "../resources/autospacing_test_fixed.txt"

WORD2VEC_FILE = "../model/autospacing/w2v.txt"
MODEL_FILE = "../model/autospacing/whitespace.ckpt"

preprocessor = Preprocessor()
if not os.path.exists(WORD2VEC_FILE):
    wordCount, trainLineCount, testLineCount = preprocessor.generateFixedLength(RAW_DATA_FILE, SEQUENCE_SIZE, TRAINING_DATE_RATE, TRAIN_FIXED_FILE, TEST_FIXED_FILE)
    print (trainLineCount, " train lines")
    W2V = preprocessor.makeW2Vfile(RAW_DATA_FILE, WORD2VEC_FILE, VECTOR_SIZE, SEQUENCE_SIZE, 0)
else:
    W2V = gensim.models.Word2Vec.load(WORD2VEC_FILE)


LEARNING_RATE = 0.01
BATCH_SIZE = 15283 # 53022
ITER_NUM = 1
DROPOUT_RATE = 0.5
EARLY_STOP_COUNT = 6

"""
#training
X_DATA, Y_DATA = preprocessor.getVectorData(TRAIN_FIXED_FILE, W2V, SEQUENCE_SIZE)
training = Training(MODEL_FILE, WORD2VEC_FILE, LEARNING_RATE, BATCH_SIZE, ITER_NUM, SEQUENCE_SIZE, VECTOR_SIZE, DROPOUT_RATE, EARLY_STOP_COUNT)
training.train(X_DATA, Y_DATA)
"""


sentence = "그책에는이별이야기가있을까어쩌면네가지금막귀퉁이를접고있는페이지에"
answer = "그 책에는 이별 이야기가 있을까 어쩌면 네가 지금 막 귀퉁이를 접고 있는 페이지에 "
X_DATA = preprocessor.getXVectorData(sentence, W2V, SEQUENCE_SIZE)
predicting = Training(MODEL_FILE, WORD2VEC_FILE, LEARNING_RATE, 1, ITER_NUM, SEQUENCE_SIZE, VECTOR_SIZE, drop_out_rate=1.0)
predicted_sentence = predicting.predict(X_DATA, sentence)
print(predicted_sentence)
if predicted_sentence == answer:
    print ("O")
else:
    print ("X")

