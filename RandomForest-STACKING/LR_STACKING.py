# coding:utf-8
import pandas as pd
import numpy as np
import math
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
class Data(object):
	def __init__(self):
		self.data = None
		self.subDataset = []
	def ImportData(self,path):
		df = pd.read_csv(path)
		self.data = df.iloc[:,1:].values
#	def DivideDataset(self,p):
		
#融合学习模型
class FusionLeaner(object):
	def __init__(self):
	#初始化时初始化要用到的模型
		self.lr = LinearRegression()
		self.logistic = LogisticRegression()
		self.nb = GaussianNB()
		self.knn = KNeighborsClassifier()
		self.dt = DecisionTreeClassifier()
		self.rf = RandomForestClassifier(max_depth=2, random_state=0)
		self.trainingDataset = [];
	#对第一次学习用到的模型进行训练
	def FirstTrainingFunc(self,X,Y):
		self.logistic.fit(X,Y)
		self.nb.fit(X,Y)
		self.knn.fit(X,Y)
		self.dt.fit(X,Y)
		self.rf.fit(X,Y)
	#用第一次学习后生成的数据集进行第二次学习训练
	def SecondTrainingFunc(self):
		x = self.trainingDataset[:,:-1]
		y = self.trainingDataset[:,-1]
		self.lr.fit(x,y)
	#生成第二次学习用的数据集
	def GenerateTrainingDataset(self,x,y):
		for i in range(len(x)):
			temp = []
			temp.append(self.dt.predict(x[i]).tolist()[0])
			temp.append(self.logistic.predict(x[i]).tolist()[0])
			temp.append(self.nb.predict(x[i]).tolist()[0])
			temp.append(self.knn.predict(x[i]).tolist()[0])
			temp.append(self.rf.predict(x[i]).tolist()[0])
			temp.append(y[i])
			self.trainingDataset.append(temp)
		self.trainingDataset = np.array(self.trainingDataset)
	#使用训练好的融合模型
	def LrPridict(self,x):
		temp = []
		temp.append(self.dt.predict(x).tolist()[0])
		temp.append(self.logistic.predict(x).tolist()[0])
		temp.append(self.nb.predict(x).tolist()[0])
		temp.append(self.knn.predict(x).tolist()[0])
		temp.append(self.rf.predict(x).tolist()[0])
		return self.lr.predict(temp)
#评价模型准确率函数
def EvaluationFn(x,y,fn):
	result = []
	isRightarr = []
	print('预测值与准确值：')
	for i in range(len(x)):
		result.append(fn(x[i]).tolist()[0])
	for i in range(len(result)):
		print((result[i],y[i]))
		if (result[i]-y[i])**2 < (1e-10):
			isRightarr.append(1)
		else:
			isRightarr.append(0)
	n = np.bincount(np.array(isRightarr))
	return (n[1]/(n[1]+n[0]))
if __name__ == '__main__':
	#读入数据文件生成第一次学习的训练集
	data = Data()
	data.ImportData('abalone.data.txt')
	X1 = data.data[:,:-1]
	Y1 = data.data[:,-1]
	#从数据集中随机取出约90%作为第二次学习的训练集，剩下约10%作为测试集
	trainingIndex = [np.random.randint(0,len(X1)-1) for i in range(int(len(X1)*0.8))]
	X2 = X1[trainingIndex,:]
	Y2 = Y1[trainingIndex]
	X3 = []
	Y3 = []
	for i in range(len(X1)):
		if i in trainingIndex:
			continue
		X3.append(X1[i,:])
		Y3.append(Y1[i])
	#开始学习
	fl = FusionLeaner()
	fl.FirstTrainingFunc(X1,Y1)
	fl.GenerateTrainingDataset(X2,Y2)
	fl.SecondTrainingFunc()
	#评价模型的准确率
	Accuracy = EvaluationFn(X3,Y3,fl.LrPridict)
	print("准确率：",Accuracy)
	