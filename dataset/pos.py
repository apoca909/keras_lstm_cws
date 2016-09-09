#!/usr/bin/env python
# -*- coding: utf-8 -*-

import h5py
import json
import codecs

mappings = {
    #人民日报标注集：863标注集
            'w':    'wp',
            't':    'nt',
            'nr':   'nh',
            'nx':   'nz',
            'nn':   'n',
            'nzz':  'n',
            'Ng':   'n',
            'f':    'nd',
            's':    'nl',
            'Vg':   'v',
            'vd':   'v',
            'vn':   'v',
            'vnn':  'v',
            'ad':   'a',
            'an':   'a',
            'Ag':   'a',
            'l':    'i',
            'z':    'a',
            'mq':   'm',
            'Mg':   'm',
            'Tg':   'nt',
            'y':    'u',
            'Yg':   'u',
            'Dg':   'd',
            'Rg':   'r',
            'Bg':   'b',
            'pn':   'p',
        }

tags_863 = {
        'a' :    [0, '形容词'],
        'b' :    [1, '区别词'],
        'c' :    [2, '连词'],
        'd' :    [3, '副词'],
        'e' :    [4, '叹词'],
        'g' :    [5, '语素字'],
        'h' :    [6, '前接成分'],
        'i' :    [7, '习用语'],
        'j' :    [8, '简称'],
        'k' :    [9, '后接成分'],
        'm' :    [10, '数词'],
        'n' :    [11, '名词'],
        'nd':    [12, '方位名词'],
        'nh':    [13, '人名'],
        'ni':    [14, '团体、机构、组织的专名'],
        'nl':    [15, '处所名词'],
        'ns':    [16, '地名'],
        'nt':    [17, '时间名词'],
        'nz':    [18, '其它专名'],
        'o' :    [19, '拟声词'],
        'p' :    [20, '介词'],
        'q' :    [21, '量词'],
        'r' :    [22, '代词'],
        'u' :    [23, '助词'],
        'v' :    [24, '动词'],
        'wp':    [25, '标点'],
        'ws':    [26, '字符串'],
        'x' :    [27, '非语素字'],
    }

def genCorpusTags():
    s = ''
    features = ['b', 'm', 'e', 's']
    for tag in tags_863.keys():
        for f in features:
            s += '\'' + tag + '-' + f + '\'' + ','
    print s

corpus_tags = [
        'nh-b','nh-m','nh-e','nh-s','ni-b','ni-m','ni-e','ni-s','nl-b','nl-m','nl-e','nl-s','nd-b','nd-m','nd-e','nd-s','nz-b','nz-m','nz-e','nz-s','ns-b','ns-m','ns-e','ns-s','nt-b','nt-m','nt-e','nt-s','ws-b','ws-m','ws-e','ws-s','wp-b','wp-m','wp-e','wp-s','a-b','a-m','a-e','a-s','c-b','c-m','c-e','c-s','b-b','b-m','b-e','b-s','e-b','e-m','e-e','e-s','d-b','d-m','d-e','d-s','g-b','g-m','g-e','g-s','i-b','i-m','i-e','i-s','h-b','h-m','h-e','h-s','k-b','k-m','k-e','k-s','j-b','j-m','j-e','j-s','m-b','m-m','m-e','m-s','o-b','o-m','o-e','o-s','n-b','n-m','n-e','n-s','q-b','q-m','q-e','q-s','p-b','p-m','p-e','p-s','r-b','r-m','r-e','r-s','u-b','u-m','u-e','u-s','v-b','v-m','v-e','v-s','x-b','x-m','x-e','x-s'
    ]

def savePosInfo(path, posInfo):
    '''保存分词训练数据字典和概率'''
    print('save pos info to %s'%path)
    fd = open(path, 'w')
    (initProb, tranProb), (vocab, indexVocab) = posInfo
    j = json.dumps((initProb, tranProb))
    fd.write(j + '\n')
    for char in vocab:
        fd.write(char.encode('utf-8') + '\t' + str(vocab[char]) + '\n')
    fd.close()

def loadPosInfo(path):
    '''载入分词训练数据字典和概率'''
    print('load pos info from %s'%path)
    fd = open(path, 'r')
    line = fd.readline()
    j = json.loads(line.strip())
    initProb, tranProb = j[0], j[1]
    lines = fd.readlines()
    fd.close()
    vocab = {}
    indexVocab = [0 for i in range(len(lines))]
    for line in lines:
        rst = line.strip().split('\t')
        if len(rst) < 2: continue
        char, index = rst[0].decode('utf-8'), int(rst[1])
        vocab[char] = index
        indexVocab[index] = char
    return (initProb, tranProb), (vocab, indexVocab)

def savePosData(path, posData):
    '''保存分词训练输入样本'''
    print('save pos data to %s'%path)
    #采用hdf5保存大矩阵效率最高
    fd = h5py.File(path,'w')
    (X, y) = posData
    fd.create_dataset('X', data = X)
    fd.create_dataset('y', data = y)
    fd.close()

def loadPosData(path):
    '''载入分词训练输入样本'''
    print('load pos data from %s'%path)
    fd = h5py.File(path,'r')
    X = fd['X'][:]
    y = fd['y'][:]
    fd.close()
    return (X, y)

def sent2vec2(sent, vocab, ctxWindows = 5):

    charVec = []
    for char in sent:
        if char in vocab:
            charVec.append(vocab[char])
        else:
            charVec.append(vocab['retain-unknown'])
    #首尾padding
    num = len(charVec)
    pad = int((ctxWindows - 1)/2)
    for i in range(pad):
        charVec.insert(0, vocab['retain-padding'] )
        charVec.append(vocab['retain-padding'] )
    X = []
    for i in range(num):
        X.append(charVec[i:i + ctxWindows])
    return X

def sent2vec(sent, vocab, ctxWindows = 5):
    chars = []
    words = sent.split()
    for word in words:
        #包含两个字及以上的词
        if len(word) > 1:
            #词的首字
            chars.append(word[0] + '_b')
            #词中间的字
            for char in word[1:(len(word) - 1)]:
                chars.append(char + '_m')
            #词的尾字
            chars.append(word[-1] + '_e')
        #单字词
        else: 
            chars.append(word + '_s')

    return sent2vec2(chars, vocab, ctxWindows = ctxWindows)

def doc2vec(fname, vocab):
    '''文档转向量'''

    #一次性读入文件，注意内存
    fd = codecs.open(fname, 'r', 'utf-8')
    lines = fd.readlines()
    fd.close()

    #样本集
    X = []
    y = []

    #标注统计信息
    tagSize = len(corpus_tags)
    tagCnt = [0 for i in range(tagSize)]
    tagTranCnt = [[0 for i in range(tagSize)] for j in range(tagSize)]

    #遍历行
    for line in lines:
        #按空格分割
        words = line.strip('\n').split()
        #每行的分词信息
        chars = []
        tags = []
        #遍历词
        for word in words:
            rst = word.split('/')
            if len(rst) <= 0:
                print word
                continue
            word, tag = rst[0], rst[1].decode('utf-8')
            if tag not in tags_863:
                tag = mappings[tag]
            #包含两个字及以上的词
            if len(word) > 1:
                #词的首字
                chars.append(word[0] + '_b')
                tags.append(corpus_tags.index(tag + '-' + 'b'))
                #词中间的字
                for char in word[1:(len(word) - 1)]:
                    chars.append(char + '_m')
                    tags.append(corpus_tags.index(tag + '-' + 'm'))
                #词的尾字
                chars.append(word[-1] + '_e')
                tags.append(corpus_tags.index(tag + '-' + 'e'))
            #单字词
            else: 
                chars.append(word + '_s')
                tags.append(corpus_tags.index(tag + '-' + 's'))

        #字向量表示
        lineVecX = sent2vec2(chars, vocab, ctxWindows = 7)

        #统计标注信息
        lineVecY = []
        lastTag = -1
        for tag in tags:
            #向量
            lineVecY.append(tag)
            #lineVecY.append(corpus_tags[tag])
            #统计tag频次
            tagCnt[tag] += 1
            #统计tag转移频次
            if lastTag != -1:
                tagTranCnt[lastTag][tag] += 1
            #暂存上一次的tag
            lastTag = tag

        X.extend(lineVecX)
        y.extend(lineVecY)

    #字总频次
    charCnt = sum(tagCnt)
    #转移总频次
    tranCnt = sum([sum(tag) for tag in tagTranCnt])
    #tag初始概率
    initProb = []
    for i in range(tagSize):
        initProb.append(tagCnt[i]/float(charCnt))
    #tag转移概率
    tranProb = []
    for i in range(tagSize):
        p = []
        for j in range(tagSize):
            p.append(tagTranCnt[i][j]/float(tranCnt))
        tranProb.append(p)

    return X, y, initProb, tranProb

def vocabAddChar(vocab, indexVocab, index, char):
    if char not in vocab:
        vocab[char] = index
        indexVocab.append(char)
        index += 1
    return index

def genVocab(fname, delimiters = [' ', '\n']):

    #一次性读入文件，注意内存
    fd = codecs.open(fname, 'r', 'utf-8')
    lines = fd.readlines()
    fd.close()

    vocab = {}
    indexVocab = []
    #遍历所有行
    index = 0
    for line in lines:
        words = line.strip().split()
        if words <= 0: continue
        #遍历所有词
        for word in words:
            word, tag = word.split('/')
            #包含两个字及以上的词
            if len(word) > 1:
                #词的首字
                char = word[0] + '_b'
                index = vocabAddChar(vocab, indexVocab, index, char)
                #词中间的字
                for char in word[1:(len(word) - 1)]:
                    char = char + '_m'
                    index = vocabAddChar(vocab, indexVocab, index, char)
                #词的尾字
                char = word[-1] + '_e'
                index = vocabAddChar(vocab, indexVocab, index, char)
            #单字词
            else: 
                char = word + '_s'
                index = vocabAddChar(vocab, indexVocab, index, char)

    #加入未登陆新词和填充词
    vocab['retain-unknown'] = len(vocab)
    vocab['retain-padding'] = len(vocab)
    indexVocab.append('retain-unknown')
    indexVocab.append('retain-padding')
    #返回字典与索引
    return vocab, indexVocab

def load(fname):
    print 'train from file', fname
    delims = [' ', '\n']
    vocab, indexVocab = genVocab(fname)
    X, y, initProb, tranProb = doc2vec(fname, vocab)
    print len(X), len(y), len(vocab), len(indexVocab)
    print initProb
    print tranProb
    return (X, y), (initProb, tranProb), (vocab, indexVocab)

def test():
    load('../data/pos.train')

if __name__ == '__main__':
    test()