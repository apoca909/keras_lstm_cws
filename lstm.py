#!/usr/bin/env python
#-*- coding: utf-8 -*-

'''
bi-directional LSTM
'''
import numpy as np
import json
import h5py
import codecs

from dataset import pos
from util import viterbi

from sklearn.cross_validation import train_test_split

from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential,Graph, model_from_json
from keras.layers.core import Dense, Dropout, Activation, TimeDistributedDense
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU, SimpleRNN

from dataset import pos

def train(posInfo, posData, modelPath, weightPath):

    (initProb, tranProb), (vocab, indexVocab) = posInfo
    (X, y) = posData

    train_X, test_X, train_y, test_y = train_test_split(X, y , train_size=0.9, random_state=1)

    train_X = np.array(train_X)
    train_y = np.array(train_y)
    test_X = np.array(test_X)
    test_y = np.array(test_y)

    outputDims = len(pos.corpus_tags)
    Y_train = np_utils.to_categorical(train_y, outputDims)
    Y_test = np_utils.to_categorical(test_y, outputDims)
    batchSize = 128
    vocabSize = len(vocab) + 1
    wordDims = 100
    maxlen = 7
    hiddenDims = 100

    w2vModel, vectorSize = pos.load('model/pChar.model')
    embeddingDim = int(vectorSize)
    embeddingUnknown = [0 for i in range(embeddingDim)]
    embeddingWeights = np.zeros((vocabSize + 1, embeddingDim))
    for word, index in vocab.items():
        if word in w2vModel:
            e = w2vModel[word]
        else:
            print word
            e = embeddingUnknown
        embeddingWeights[index, :] = e

    #LSTM
    model = Sequential()
    model.add(Embedding(output_dim = embeddingDim, input_dim = vocabSize + 1, 
        input_length = maxlen, mask_zero = True, weights = [embeddingWeights]))
    model.add(LSTM(output_dim = hiddenDims, return_sequences = True))
    model.add(LSTM(output_dim = hiddenDims, return_sequences = False))
    model.add(Dropout(0.5))
    model.add(Dense(outputDims))
    model.add(Activation('softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')

    result = model.fit(train_X, Y_train, batch_size = batchSize, 
                    nb_epoch = 20, validation_data = (test_X,Y_test), show_accuracy=True)

    j = model.to_json()
    fd = open(modelPath, 'w')
    fd.write(j)
    fd.close()

    model.save_weights(weightPath)

    return model
    #Bi-directional LSTM

def loadModel(modelPath, weightPath):

    fd = open(modelPath, 'r')
    j = fd.read()
    fd.close()

    model = model_from_json(j)

    model.load_weights(weightPath)

    return model


# 根据输入得到标注推断
def posSent(sent, model, posInfo):
    (initProb, tranProb), (vocab, indexVocab) = posInfo
    vec = pos.sent2vec(sent, vocab, ctxWindows = 7)
    vec = np.array(vec)
    probs = model.predict_proba(vec)
    #classes = model.predict_classes(vec)

    prob, path = viterbi.viterbi(vec, pos.corpus_tags, initProb, tranProb, probs.transpose())

    ss = ''
    words = sent.split()
    index = -1
    for word in words:
        for char in word:
            index += 1
        ss += word + '/' + pos.tags_863[pos.corpus_tags[path[index]][:-2]][1].decode('utf-8') + ' '
        #ss += word + '/' + pos.corpus_tags[path[index]][:-2] + ' '

    return ss[:-1]

def posFile(fname, dstname, model, posInfo):
    fd = codecs.open(fname, 'r', 'utf-8')
    lines = fd.readlines()
    fd.close()

    fd = open(dstname, 'w')
    for line in lines:
        rst = posSent(line.strip(), model, posInfo)
        fd.write(rst.encode('utf-8') + '\n')
    fd.close()

def test():
    print 'Loading vocab...'
    #(X, y), (initProb, tranProb), (vocab, indexVocab) = pos.load('data/pos.train')
    #posInfo = ((initProb, tranProb), (vocab, indexVocab))
    #posData = (X, y)
    #pos.savePosInfo('./model/pos.info', posInfo)
    #pos.savePosData('./model/pos.data', posData)
    posInfo = pos.loadPosInfo('./model/pos.info')
    posData = pos.loadPosData('./model/pos.data')
    print 'Done!'
    print 'Loading model...'
    #model = train(posInfo, posData, './model/pos.w2v.model', './model/pos.w2v.model.weights')
    model = loadModel('./model/pos.w2v.model', './model/pos.w2v.model.weights')
    #model = loadModel('./model/pos.model', './model/pos.model.weights')
    print 'Done!'
    print '-------------start predict----------------'
    s = u'为 寂寞 的 夜空 画 上 一个 月亮'
    print posSent(s, model, posInfo)
    #posFile('~/work/corpus/icwb2/testing/msr_test.utf8', './msr_test.utf8.pos', model, posInfo)

if __name__ == '__main__':
    test()