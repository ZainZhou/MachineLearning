# coding:utf-8
import pandas as pd
import numpy as np
import math
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
class Data(object):
	def __init__(self):
		self.data = None
		self.classes = None
		self.subDataset = []
	def ImportData(self,path):
		df = pd.read_csv(path)
		self.data = df.iloc[:,1:].values
class FusionLeaner(object):
	def __init__(self):
		self.lr = LinearRegression()
		self.logistic = LogisticRegression()
		self.nb = GaussianNB()
		self.knn = KNeighborsClassifier()
		self.dt = DecisionTreeClassifier()
		self.rf = RandomForestClassifier(max_depth=3, random_state=0)
		self.trainingDataset = [];
	def FirstTrainingFunc(self,X,Y):
		self.logistic.fit(X,Y)
		self.nb.fit(X,Y)
		self.knn.fit(X,Y)
		self.dt.fit(X,Y)
		self.rf.fit(X,Y)
	def SecondTrainingFunc(self):
		x = self.trainingDataset[:,:-1]
		y = self.trainingDataset[:,-1]
		self.lr.fit(x,y)
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
	def LrPridict(self,x):
		temp = []
		temp.append(self.dt.predict(x).tolist()[0])
		temp.append(self.logistic.predict(x).tolist()[0])
		temp.append(self.nb.predict(x).tolist()[0])
		temp.append(self.knn.predict(x).tolist()[0])
		temp.append(self.rf.predict(x).tolist()[0])
		return self.lr.predict(temp)
if __name__ == '__main__':
	data = Data()
	data.ImportData('abalone.data.txt')
	X1 = data.data[:,:-1]
	Y1 = data.data[:,-1]
	trainingIndex = [np.random.randint(0,len(X1)-1) for i in range(int(len(X1)/3))]
	X2 = X1[trainingIndex,:]
	Y2 = Y1[trainingIndex]
	fl = FusionLeaner()
	fl.FirstTrainingFunc(X1, Y1)
	fl.GenerateTrainingDataset(X2, Y2)
	fl.SecondTrainingFunc()
	print(fl.LrPridict([0.56,0.44,0.135,0.8025,0.35,0.1615,0.259]))
	
	