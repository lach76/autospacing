import os
import gensim
from preprocessing import Preprocessor
from training import Training

MAX_SEQUENCE_SIZE = 30
TRAINING_DATE_RATE = 0.8
SEQUENCE_SIZE = 50
VECTOR_SIZE = 100  # vocaburary size
RAW_DATA_FILE = "resources/poems.txt"
WORD2VEC_FILE = "model/w2v.txt"

TRAIN_FIXED_FILE = "resources/train_fixed.txt"
TEST_FIXED_FILE = "resources/test_fixed.txt"


preprocessor = Preprocessor()
if not os.path.exists(WORD2VEC_FILE):
    preprocessor.generateFixedLength(RAW_DATA_FILE,
                                     SEQUENCE_SIZE,
                                     TRAINING_DATE_RATE,
                                     TRAIN_FIXED_FILE,
                                     TEST_FIXED_FILE)
    W2V = preprocessor.makeW2Vfile(RAW_DATA_FILE, WORD2VEC_FILE, VECTOR_SIZE, SEQUENCE_SIZE, 0)
else:
    W2V = gensim.models.Word2Vec.load(WORD2VEC_FILE)


LEARNING_RATE = 0.01
BATCH_SIZE = 2059
ITER_NUM = 5
DROPOUT_RATE = 0.7
EARLY_STOP_COUNT = 3

""" training
X_DATA, Y_DATA = preprocessor.getVectorData(TRAIN_FIXED_FILE, W2V, SEQUENCE_SIZE)
training = Training(LEARNING_RATE, BATCH_SIZE, ITER_NUM, SEQUENCE_SIZE, VECTOR_SIZE, DROPOUT_RATE, EARLY_STOP_COUNT)
training.train(X_DATA, Y_DATA)
"""

sentence = "그책에는이별이야기가있을까어쩌면네가지금막귀퉁이를접고있는페이지에"
X_DATA = preprocessor.getXVectorData(sentence, W2V, SEQUENCE_SIZE)
predicting = Training(LEARNING_RATE, 1, ITER_NUM, SEQUENCE_SIZE, VECTOR_SIZE, drop_out_rate=1.0)
predicted_sentence = predicting.predict(X_DATA, sentence)
print(predicted_sentence)