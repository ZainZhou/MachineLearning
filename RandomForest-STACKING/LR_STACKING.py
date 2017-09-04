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
		self.lr.fit(self.trainingDataset[:,:-1],self.trainingDataset[:,-1])
	def GenerateTrainingDataset(self,x,y):
		for dt in x:
			temp = []
			temp.append(self.dt.predict(dt))
			temp.append(self.logistic.predict(dt))
			temp.append(self.nb.predict(dt))
			temp.append(self.knn.predict(dt))
			temp.append(self.rf.predict(dt))
			temp.append(y[i])
			self.trainingDataset.append(temp)
	def LrPridict(self,x):
		return self.lr.predict(x)
		
if __name__ == '__main__':
	data = Data()
	data.ImportData('abalone.data.txt')
	X = data.data[:,:-1]
	Y = data.data[:,-1]
	
	