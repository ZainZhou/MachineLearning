from math import log
import operator
#香农熵计算
def ShannonEntropy(dataSet):
    labelCounts = {} #记录标签出现次数
    dataNum = len(dataSet) #数据条数
    for data in dataSet:
        cLabel = data[-1]
        if cLabel not in labelCounts.keys():
            labelCounts[cLabel] = 0
        labelCounts[cLabel] += 1
    SE = 0.0
    for key in labelCounts:
        percent = float(labelCounts[key])/dataNum
        SE -= percent*log(percent,2)
    return SE
#划分数据集
def DiviDataSet(dataSet, feature, v):
    dividDataSet = []
    for data in dataSet:
        if data[feature] == v:
            tempDataSet = data[:feature]
            tempDataSet.extend(data[feature+1:])
            dividDataSet.append(tempDataSet)
    return dividDataSet
#选择根节点属性来划分数据集
def selectBestFeature(dataSet):
    numAttr = len(dataSet[0]) - 1
    baseEnt = ShannonEntropy(dataSet)
    bestInfoInc = 0.0
    bestFeature = -1
    for i in range(numAttr):
        uValues = set([example[i] for example in dataSet])
        newEnt = 0.0
        for value in uValues:
            subDataSet = DiviDataSet(dataSet, i, value)
            percent = len(subDataSet)/float(len(dataSet))
            newEnt += percent * ShannonEntropy(subDataSet)
            InfoInc = baseEnt - newEnt
        if (InfoInc > bestInfoInc):
            bestInfoInc = InfoInc
            bestFeature = i
    return bestFeature
#计算出现次数最多的分类
def majorityCnt(classList):
    classCount = {}
    for v in classList:
        if v not in classCount.keys():
            classCount[v] = 0
        classCount[v] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]
#建树
def createTree(dataSet,labels,n):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = selectBestFeature(dataSet)
    bestFeatLabel = labels[bestFeat]
    if bestFeatLabel not in FeatureSubset:
        if len(FeatureSubset) == n:
            return
        FeatureSubset.append(bestFeatLabel)
    DTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        DTree[bestFeatLabel] [value] = createTree(DiviDataSet(dataSet, bestFeat, value),subLabels,n)
    return DTree
dataSet = []
with open('dataset/car.data.txt') as f:
    for x in f.readlines():
        dataSet.append(x.strip().split(','))
labels = ['buying','maint','doors','persons','lug_boot','safety']
FeatureSubset = []
DTree = createTree(dataSet,labels,4)
print(FeatureSubset)