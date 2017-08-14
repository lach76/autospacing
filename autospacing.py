from preprocessing import Preprocessor
from training import Training

MAX_SEQUENCE_SIZE = 50
TRAINING_DATE_RATE = 0.8
SEQUENCE_SIZE = 50
VECTOR_SIZE = 100  # vocaburary size
RAW_DATA_FILE = "resources/poems.txt"
WORD2VEC_FILE = "model/w2v.txt"

TRAIN_FIXED_FILE = "resources/train_fixed.txt"
TEST_FIXED_FILE = "resources/test_fixed.txt"


preprocessor = Preprocessor()
preprocessor.generateFixedLength(RAW_DATA_FILE,
                                 SEQUENCE_SIZE,
                                 TRAINING_DATE_RATE,
                                 TRAIN_FIXED_FILE,
                                 TEST_FIXED_FILE)


W2V = preprocessor.makeW2Vfile(RAW_DATA_FILE, WORD2VEC_FILE, VECTOR_SIZE, SEQUENCE_SIZE, 0)
X_DATA, Y_DATA = preprocessor.getVectorData(TRAIN_FIXED_FILE, W2V, SEQUENCE_SIZE)


LEARNING_RATE = 0.01
BATCH_SIZE = 100
DROPOUT_RATE = 0.7
training = Training(LEARNING_RATE, BATCH_SIZE, SEQUENCE_SIZE, VECTOR_SIZE, DROPOUT_RATE)
training.train(X_DATA, Y_DATA)