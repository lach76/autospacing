"""
    입력 파일 전처리
"""
import random
import gensim
import numpy as np

class Preprocessor(object):

    # Training 파일을 읽어서 Char2Vector 생성
    def generateFixedLength(self, inFile, sequenceSize, trainingDataRate, outTrainFile, outTestFile):
        padd = ' '
        trainLineCount = 0
        testLineCount = 0
        wordCount = 0

        infp = open(inFile, 'r')
        lines = []
        lineCount = 0
        while True:
            line = infp.readline().strip('\n')
            if not line: break
            while True:
                if (line.count("\"") < 1): break
                else: line = line.replace("\"", "")
            if line[1:]=='NAME': continue
            if len(line) > sequenceSize: continue
            #if (lineCount >= maxDataSize): break

            line = line.strip()
            lines.append(line)
            if wordCount < len(line): wordCount = len(line)

            lineCount += 1
        infp.close()

        random.shuffle(lines)
        train_data = lines[:int((len(lines)) * trainingDataRate)]  # Remaining 80% to training set
        test_data = lines[int(len(lines) * trainingDataRate):]  # Splits 20% data to test set

        train_size = int(len(lines) * trainingDataRate)
        test_size = int(len(lines) * (1 - trainingDataRate))

        # trainLineCount
        outTrainFp = open(outTrainFile, 'w')
        sentence = ""
        aline = "";
        alabel = "";
        index = 0
        isBreak = False
        for i in range(0, len(train_data)):
            line = train_data[i].strip()
            if (sentence == ""):
                sentence = line
            else:
                sentence = sentence + " " + line

            size = len(sentence)
            for i in range(0, size-1):
                if (i < size - 1):
                    if (sentence[0] != ' '):
                        if sentence[1] == ' ':
                            alabel = alabel + "1"
                        else:
                            alabel = alabel + "0"
                        aline = aline + sentence[0]
                        index = index + 1
                    sentence = sentence[1:]

                if (index == sequenceSize):
                    outTrainFp.write(aline)
                    outTrainFp.write("\t")
                    outTrainFp.write(alabel)
                    outTrainFp.write("\n")

                    index = 0
                    aline = ""
                    alabel = ""
                    trainLineCount = trainLineCount + 1
                    if trainLineCount >= train_size:
                        isBreak = True
                        break;
            if isBreak:
                break;

        if (isBreak == False and index < sequenceSize):
            size = len(sentence)
            for i in range(0, size-1):
                if (sentence[i+1] != ' '):
                    if sentence[i] == ' ':
                        alabel = alabel + "1"
                    else:
                        alabel = alabel + "0"
                    aline = aline + sentence[i]
                    index = index + 1

            alabel = alabel + "1"
            aline = aline + sentence[size-1]
            index = index + 1

            diff = sequenceSize - index
            if diff > 0 : # add padding
                alabel += '0' * diff
                aline += padd * diff

            outTrainFp.write(aline)
            outTrainFp.write("\t")
            outTrainFp.write(alabel)
            outTrainFp.write("\n")
            trainLineCount = trainLineCount + 1

        outTrainFp.close()

        # testLineCount
        outTestFp = open(outTestFile, 'w')
        sentence = ""
        aline = "";
        alabel = "";
        index = 0
        isBreak = False
        for i in range(0, len(test_data)):
            line = test_data[i].strip()
            if (sentence == ""):
                sentence = line
            else:
                sentence = sentence + " " + line

            size = len(sentence)
            for i in range(0, size - 1):
                if (i < size - 1):
                    if (sentence[0] != ' '):
                        if sentence[1] == ' ':
                            alabel = alabel + "1"
                        else:
                            alabel = alabel + "0"
                        aline = aline + sentence[0]
                        index = index + 1
                    sentence = sentence[1:]

                if (index == sequenceSize):
                    outTestFp.write(aline)
                    outTestFp.write("\t")
                    outTestFp.write(alabel)
                    outTestFp.write("\n")

                    index = 0
                    aline = ""
                    alabel = ""
                    testLineCount = testLineCount + 1
                    if testLineCount > test_size:
                        isBreak = True
                        break;
            if isBreak:
                break;

        if (isBreak == False and index < sequenceSize):
            size = len(sentence)
            for i in range(0, size - 1):
                if (sentence[i + 1] != ' '):
                    if sentence[i] == ' ':
                        alabel = alabel + "1"
                    else:
                        alabel = alabel + "0"
                    aline = aline + sentence[i]
                    index = index + 1

            alabel = alabel + "1"
            aline = aline + sentence[size - 1]
            index = index + 1

            diff = sequenceSize - index
            if diff > 0:  # add padding
                alabel += '0' * diff
                aline += padd * diff

            outTestFp.write(aline)
            outTestFp.write("\t")
            outTestFp.write(alabel)
            outTestFp.write("\n")
            testLineCount = testLineCount + 1
        outTrainFp.close()

        return wordCount, trainLineCount, testLineCount

    def generatePassThrough(self, inFile, maxSequenceLength, maxDataSize, trainingDataRate, outTrainFile, outTestFile):
        wordCount = 0
        trainLineCount = 0
        testLineCount = 0

        lines = []
        infp = open(inFile, 'r')
        lineCount = 0
        while True:
            line = infp.readline().strip('\n')
            if not line: break
            while True:
                if (line.count("\"") < 1): break
                else: line = line.replace("\"", "")
            if line[1:]=='NAME': continue
            if len(line) > maxSequenceLength: continue
            if (lineCount >= maxDataSize): break

            line = line.strip()
            lines.append(line)
            if wordCount < len(line) : wordCount = len(line)
            lineCount += 1
        infp.close()

        random.shuffle(lines)
        train_data = lines[:int((len(lines)) * trainingDataRate)]  # Remaining 80% to training set
        test_data = lines[int(len(lines) * trainingDataRate):]  # Splits 20% data to test set

        outTrainFp = open(outTrainFile, 'w')
        aline = "";
        alabel = "";
        for i in range(0, len(train_data)):
            line = train_data[i]
            size = len(line)
            for i in range(0, size-1):
                if (line[i] != ' '):
                    if line[i+1] == ' ':
                        alabel = alabel + "1"
                    else:
                        alabel = alabel + "0"
                    aline = aline + line[i]

            alabel = alabel + "1"
            aline = aline + line[size-1]

            outTrainFp.write(aline)
            outTrainFp.write("\t")
            outTrainFp.write(alabel)
            outTrainFp.write("\n")
            trainLineCount = trainLineCount + 1

            aline = ""
            alabel = ""
        outTrainFp.close()

        outTestFp = open(outTestFile, 'w')
        aline = "";
        alabel = "";
        for i in range(0, len(test_data)):
            line = test_data[i]
            size = len(line)
            for i in range(0, size - 1):
                if (line[i] != ' '):
                    if line[i + 1] == ' ':
                        alabel = alabel + "1"
                    else:
                        alabel = alabel + "0"
                    aline = aline + line[i]

            alabel = alabel + "1"
            aline = aline + line[size - 1]

            outTestFp.write(aline)
            outTestFp.write("\t")
            outTestFp.write(alabel)
            outTestFp.write("\n")
            testLineCount = testLineCount + 1

            aline = ""
            alabel = ""
        outTestFp.close()

        return wordCount, trainLineCount, testLineCount


    def makeW2Vfile(self, inFile, outFile, vectorSize, sequenceSize, sg=1):
        # Training 파일을 읽어서 Char2Vector 생성
        line_count = 0
        label = 0
        sentences = []
        # train 파일
        infp = open(inFile, 'r')
        while True:
            line = infp.readline().strip('\n')
            if not line: break
            while True:
                if (line.count("\"") < 1): break
                else: line = line.replace("\"", "")
            if line[1:]=='NAME': continue


            line = line.strip()
            umjuls = [w for w in line]
            sentences.append(umjuls)
            line_count = line_count + 1
        infp.close()

        w2v = gensim.models.Word2Vec(sentences, size=vectorSize, window=sequenceSize, min_count=1, workers=4, sg=sg)
        w2v.save(outFile)

        return w2v

    # 데이터 + padding
    def pad_Xsequences_train(self, sentences, maxlen):
        nullItem = []
        itemLen = len(sentences[0][0])
        for i in range(0, itemLen):
            nullItem.append(0.0)

        newVec = []
        len1 = len(sentences)
        for i in range(0, len1):
            vec = []
            len2 = len(sentences[i])
            for j in range(0, len2):
                vec.append(sentences[i][j])
                j += 1
            for j in range(0, maxlen - len2):
                vec.append(nullItem)
                j += 1
            newVec.append(vec)
            i += 1

        return np.array(newVec, dtype='f')

    # padding + 데이터
    def pad_Xsequences_test(self, sentences, maxlen):
        nullItem = []
        itemLen = len(sentences[0][0])
        for i in range(0, itemLen):
            nullItem.append(0.0)

        newVec = []
        len1 = len(sentences)
        for i in range(0, len1):
            vec = []
            len2 = len(sentences[i])
            for j in range(0, maxlen - len2):
                vec.append(nullItem)
                j += 1
            for j in range(0, len2):
                vec.append(sentences[i][j])
                j += 1
            newVec.append(vec)
            i += 1

        return np.array(newVec, dtype='f')

    # 데이터 + padding
    def pad_Ysequences_train(self, label, maxlen):
        len1 = len(label)
        newVec = []
        nullItem = 0
        for i in range(0, len1):
            vec = []
            len2 = len(label[i])
            for j in range(0, len2):
                vec.append(label[i][j])
                j += 1
            for j in range(0, maxlen - len2):
                vec.append(nullItem)
                j += 1
            newVec.append(vec)
            i += 1

        return np.array(newVec, dtype='int32')

    # padding + 데이터
    def pad_Ysequences_test(self, label, maxlen):
        len1 = len(label)
        newVec = []
        nullItem = 0
        for i in range(0, len1):
            vec = []
            len2 = len(label[i])
            for j in range(0, maxlen - len2):
                vec.append(nullItem)
                j += 1
            for j in range(0, len2):
                vec.append(label[i][j])
                j += 1
            newVec.append(vec)
            i += 1

        return np.array(newVec, dtype='int32')


    def getVectorData(self, inFilename, w2vModel, sequence_size):
        count = 0
        sentences = []
        labels = []

        f = open(inFilename, 'r')
        while True:
            line = f.readline().strip('\n')
            if not line: break

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

        inputX = self.pad_Xsequences_test(sentences, sequence_size)
        inputY = self.pad_Ysequences_test(labels, sequence_size)

        return inputX, inputY

    def getXVectorData(self, line, w2vModel, sequence_size):
        sentence = [w2vModel.wv[w] for w in line]
        sentences = []
        sentences.append(sentence)
        inputX = self.pad_Xsequences_test(sentences, sequence_size)
        return inputX