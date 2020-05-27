# -*- coding: utf-8 -*-
import operator
from math import log

def createDataSet():
    #dataSet的三列，特征1，特征2，判断结果
    dataSet = [[1,1,'yes'],
               [1,1,'yes'],
               [1,0,'no'],
               [0,1,'no'],
               [0,1,'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet,labels

def calcShannonEnt(dataSet):
    numEntries = len(dataSet) #读取dataSet总长度，数据量
    labelCounts = {} #给判断结果出一个词典，key为结果，value为结果数量
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
            #如果没有出现过这个结果，就预设一个初始值为0的组
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts :
        prob = float(labelCounts[key])/numEntries #浮点小数
        shannonEnt -= prob*log(prob,2)
        #熵越高，混合的信息量越多
    return shannonEnt

def splitDataSet(dataSet, axis, value):
    #分组方法
    retDataSet = [] #新组
    for featVec in dataSet:
        if featVec[axis] == value: #如果一条数据的axis位 等于 value
            reducedFeatVec = featVec[:axis] #切割
            reducedFeatVec.extend(featVec[axis+1:]) #加入
            retDataSet.append(reducedFeatVec)
            #总之效果就是该元素满足条件才能被选入，选入之后，筛去该元素建立新的dict
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet] # 每次读取同一类型特征值中的全部
        uniqueVals = set(featList) # set是几何集合函数，是无序不重复的，等于去除重复相同的特征值，留下不同的
        newEntropy = 0.0 # 所求熵的初始值
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            # 原始数据集中的i位满足value值，进行筛选分类 留下满足项
            # 所有不同value都要经历分类完之后计算熵的过程
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
            # 带上概率系数把这个划分方案的熵总和求出
            infoGain = baseEntropy - newEntropy #新熵和旧熵的差为信息增益
        if (infoGain > bestInfoGain):
            # 找到最好的条件熵 信息增益越大越好
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt((classList))
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
    return myTree


