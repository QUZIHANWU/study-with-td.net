#!/usr/bin/env python
#-*- coding:UTF-8 -*-（添加）

from __future__ import print_function
from numpy import *

def textParse(bigString):
    import re
    # 使用正则表达式来切分句子，其中分隔符是除单词、数字外的任意字符串
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def spamTest():

    docList = []
    classList = []
    fullText = []
    for i in range(1, 26):
        # 切分，解析数据，并归类为 1 类别
        wordList = textParse(open('/Users/quzihanwu/Desktop/1.rtf' % i).read())
        docList.append(wordList)
        classList.append(1)
        # 切分，解析数据，并归类为 0 类别
        wordList = textParse(open('/Users/quzihanwu/Desktop/1.rtf' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    # 创建词汇表    
    vocabList = createVocabList(docList)
    trainingSet = range(50)
    testSet = []
    # 随机取 10 个邮件用来测试
    for i in range(10):
        # random.uniform(x, y) 随机生成一个范围为 x - y 的实数
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print('the errorCount is: ', errorCount)
    print('the testSet length is :', len(testSet))
    print('the error rate is :', float(errorCount)/len(testSet))

#def testParseTest():
    #print(textParse(open('/Users/quzihanwu/Desktop/1.rtf').read()))

print(textParse(open('/Users/quzihanwu/Desktop/1.rtf').read()))
#print(spamTest(open('/Users/quzihanwu/Desktop/1.rtf').read()))