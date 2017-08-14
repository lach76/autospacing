import random

# Training 파일을 읽어서 Char2Vector 생성
def generate(rawDataFile, trainFile, testFile, maxDataSize):

    infp = open(rawDataFile, 'r')
    lines = []
    firstLine = True
    while True:
        line = infp.readline().strip('\n')
        if (firstLine):
            firstLine = False
            continue

        if not line: break

        while True:
            if (line.count("\"") < 1): break
            else: line = line.replace("\"", "")

        while True:
            if (line.count("\t") < 1): break
            else: line = line.replace("\t", "")

        line = line.strip()
        lines.append(line)
    infp.close()

    random.shuffle(lines)
    train_data = lines[0:maxDataSize]
    test_data = lines[maxDataSize:2*maxDataSize]

    outTrainFp = open(trainFile, 'w')
    for i in range(0, len(train_data)):
        line = train_data[i].strip()
        outTrainFp.write(line)
        outTrainFp.write('\n')
    outTrainFp.close()

    outTestFp = open(testFile, 'w')
    for i in range(0, len(test_data)):
        line = test_data[i].strip()
        outTestFp.write(line)
        outTestFp.write('\n')
    outTestFp.close()

if __name__ == "__main__":
    MAX_DATA_SIZE = 2000

    #raw_data_file = "rsc/all_products.txt"
    raw_data_file = "rsc/all_product_data.txt"

    train_file = "rsc/train.txt"
    test_file = "rsc/test.txt"

    generate(raw_data_file, train_file, test_file, MAX_DATA_SIZE)

