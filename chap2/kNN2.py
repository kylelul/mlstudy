# -*- coding: utf-8 -*-
from numpy import *
import operator

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis = 1)
    distances = sqDistances**0.5
    sortedDistIndices = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndices[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0)
    sortedClassCount = sorted(classCount.iteritems(),
                              key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines,3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector

def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals,(m,1))
    normDataSet = normDataSet/tile(ranges,(m,1))
    return normDataSet, ranges, minVals

def datingClassTest():
    hoRatio = 0.10
    #file2matrix 读取数值
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
    #autoNorm归一化特征值
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    #取0.1部分的为测试部分，剩下0.1-1的部分都是训练集
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        #classify0的返回值是分类器鉴别测试值属于哪一类
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:], datingLabels[numTestVecs:m],3)
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult,datingLabels[i])
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0 #错误计数
    print "the total error rate is: %f" % (errorCount/float(numTestVecs))
def classifyPerson():
    resultList = ['not at all','in small doses','in large doses']
    percentTats = float(raw_input("percentage of time spent playing video games?"))
    ffMiles = float(raw_input("frequent flier miles earned per year?"))
    iceCream = float(raw_input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
    print "You will probably like this person: ", resultList[classifierResult - 1]

def img2vector(filename):
    #上面的数据训练方法是对应一维向量的，所以要把32*32的图片信息转进1x1024的向量
    returnVect = zeros((1,1024)) #一行，长度为1024的，全0向量
    fr = open(filename)
    for i in range(32):
        # i 0-31 行 逐行阅读
        lineStr = fr.readline()
        # j 0-31 每一行逐列阅读
        for j in range(32):
            # 填充1024向量，0-31、32-63.... ，填充数字
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():
    hwLabels = []
    # .listdir() 读取该文件夹里的所有文件
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    # 行数和文件数一致的零矩阵
    trainingMat = zeros((m,1024))
    for i in range(m):






