import tensorflow as tf
import numpy as np
##The relationship between subject and object is irreversible.

class2label = {
               'RP(subject,object)': 0, 'LA(subject,object)': 1,
               'HB(subject,object)': 2, 'PN(subject,object)': 3,
               'NAT(subject,object)': 4, 'AL(subject,object)': 5,
               'INVEN(subject,object)': 6, 'GRA(subject,object)': 7,
               'HQ(subject,object)': 8, 'SP(subject,object)': 9,
               'ACT(subject,object)': 10, 'CM(subject,object)': 11,
               'AT(subject,object)': 12, 'DY(subject,object)': 13,
               'DIR(subject,object)': 14, 'ABBR(subject,object)': 15,
               'CP(subject,object)': 16, 'RG(subject,object)': 17,
               'BP(subject,object)': 18, 'PA(subject,object)': 19,
               'AS(subject,object)': 20, 'CITY(subject,object)': 21,
               'RS(subject,object)': 22, 'FAR(subject,object)': 23,
               'PRESS(subject,object)': 24, 'OL(subject,object)': 25,
               'HOST(subject,object)': 26, 'HEIG(subject,object)': 27,
               'WIFE(subject,object)': 28, 'CLI(subject,object)': 29,
               'BO(subject,object)': 30, 'SING(subject,object)': 31,
               'SD(subject,object)': 32, 'LYR(subject,object)': 33,
               'WEB(subject,object)': 34, 'AP(subject,object)': 35,
               'AREA(subject,object)': 36, 'MOT(subject,object)': 37,
               'PC(subject,object)': 38, 'WC(subject,object)': 39,
               'CN(subject,object)': 40, 'MB(subject,object)': 41,
               'PCODE(subject,object)': 42, 'FP(subject,object)': 43,
               'BD(subject,object)': 44, 'GUEST(subject,object)': 45,
               'NA(subject,object)': 46, 'BDATE(subject,object)': 47,
               'WRITER(subject,object)': 48, 'OTHER(subject,object)': 49}

label2class = {
               0: 'RP(subject,object)', 1: 'LA(subject,object)',
               2: 'HB(subject,object)', 3: 'PN(subject,object)',
               4: 'NAT(subject,object)', 5: 'AL(subject,object)',
               6: 'INVEN(subject,object)', 7: 'GRA(subject,object)',
               8: 'HQ(subject,object)', 9: 'SP(subject,object)',
               10: 'ACT(subject,object)', 11: 'CM(subject,object)',
               12: 'AT(subject,object)', 13: 'DY(subject,object)',
               14: 'DIR(subject,object)', 15: 'ABBR(subject,object)',
               16: 'CP(subject,object)', 17: 'RG(subject,object)',
               18: 'BP(subject,object)', 19: 'PA(subject,object)',
               20: 'AS(subject,object)', 21: 'CITY(subject,object)',
               22: 'RS(subject,object)', 23: 'FAR(subject,object)',
               24: 'PRESS(subject,object)', 25: 'OL(subject,object)',
               26: 'HOST(subject,object)', 27: 'HEIG(subject,object)',
               28: 'WIFE(subject,object)', 29: 'CLI(subject,object)',
               30: 'BO(subject,object)', 31: 'SING(subject,object)',
               32: 'SD(subject,object)', 33: 'LYR(subject,object)',
               34: 'WEB(subject,object)', 35: 'AP(subject,object)',
               36: 'AREA(subject,object)', 37: 'MOT(subject,object)',
               38: 'PC(subject,object)', 39: 'WC(subject,object)',
               40: 'CN(subject,object)', 41: 'MB(subject,object)',
               42: 'PCODE(subject,object)', 43: 'FP(subject,object)',
               44: 'BD(subject,object)', 45: 'GUEST(subject,object)',
               46: 'NA(subject,object)', 47: 'BDATE(subject,object)',
               48: 'WRITER(subject,object)', 49: 'OTHER(subject,object)'}


def initializer():
    return tf.keras.initializers.glorot_normal()


def load_word2vec(word2vec_path, embedding_dim, vocab):
    # initial matrix with random uniform
    initW = np.random.randn(len(vocab.vocabulary_), embedding_dim).astype(np.float32) * np.sqrt(2.0 / len(vocab.vocabulary_))
    # load any vectors from the word2vec
    print("Load word2vec file {0}".format(word2vec_path))
    with open(word2vec_path, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in range(vocab_size):
            word = []
            while True:
                ch = f.read(1).decode('latin-1')
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            idx = vocab.vocabulary_.get(word)
            if idx != 0:
                initW[idx] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)
    return initW


def load_baidubaike(word2vec_path, embedding_dim, vocab):
    # initial matrix with random uniform
    initW = np.random.randn(len(vocab.vocabulary_), embedding_dim).astype(np.float32) * np.sqrt(2.0 / len(vocab.vocabulary_))
    # load any vectors from the word2vec
    print("Load baidubaike file {0}".format(word2vec_path))
    f = open(word2vec_path, 'r', encoding='utf8')
    for line in f:
        line = line.strip()
        splitLine = line.split(' ')
        word = splitLine[0]
        embedding = np.asarray(splitLine[1:], dtype='float32')
        idx = vocab.vocabulary_.get(word)
        if idx != 0:
            initW[idx] = embedding
    return initW


def load_glove(word2vec_path, embedding_dim, vocab):
    # initial matrix with random uniform
    initW = np.random.randn(len(vocab.vocabulary_), embedding_dim).astype(np.float32) * np.sqrt(2.0 / len(vocab.vocabulary_))
    # load any vectors from the word2vec
    print("Load glove file {0}".format(word2vec_path))
    f = open(word2vec_path, 'r', encoding='utf8')
    for line in f:
        splitLine = line.split(' ')
        word = splitLine[0]
        embedding = np.asarray(splitLine[1:], dtype='float32')
        idx = vocab.vocabulary_.get(word)
        if idx != 0:
            initW[idx] = embedding
    return initW

def load_bert(bert_path, embedding_dim, vocab):
    # initial matrix with random uniform
    initW = np.random.randn(len(vocab.vocabulary_), embedding_dim).astype(np.float32) * np.sqrt(2.0 / len(vocab.vocabulary_))
    # load any vectors from the bert word vector
    print("Load bert vector file {0}".format(bert_path))
    f = open(bert_path, 'r', encoding='utf8')
    for line in f:
        splitLine = line.split(' ')
        word = splitLine[0]
        embedding = np.asarray(splitLine[1:], dtype='float32')
        idx = vocab.vocabulary_.get(word)
        if idx != 0:
            initW[idx] = embedding
    return initW
