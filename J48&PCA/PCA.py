import matplotlib.pyplot as plt
import numpy as np
#协方差矩阵及其特征向量和特征值计算函数
def CovFeatureVecsVals(X):
  mean_vec = np.mean(X, axis=0)
  #归一化
  for i in range(len(X)):
    X[i,:] = (X[i,:] - mean_vec[:])
  cov_mat = np.cov(X.T)
  eig_vals, eig_vecs = np.linalg.eig(cov_mat)
  #组成特征值和特征向量对(通过numpy.linalg.eig()计算出的特征值和特征向量是从大到小对应排好序的.)
  eig_pairs = [(eig_vals[i], eig_vecs[:,i]) for i in range(len(eig_vals))]
  return eig_pairs
  
#降维矩阵计算函数：
def ReduceDimension(eig_pairs,d):
  matrix_w = np.hstack(
    (eig_pairs[i][1].reshape(4,1) for i in range(d))
  )
  return matrix_w                
#PCA降维调度函数
def RunPCA(data,d):
  X = np.array(data,dtype = 'float64')
  d = int(d)
  if d >= X.shape[1] or d <= 0:
    return 0
  #计算协方差矩阵及其特征向量和特征值
  eig_pairs = CovFeatureVecsVals(X)
  #计算降维矩阵
  matrix_w = ReduceDimension(eig_pairs,d)
  #降维
  Y = X.dot(matrix_w)
  return Y

  
if __name__ == '__main__':
  #导入数据
  data = []
  with open('dataset/iris.data') as f:
    for i in f.readlines():
      data.append(i.strip().split(',')[:-1])
  #运行PCA调度函数
  Y = RunPCA(data,2)
  if type(Y) == type(0):
      print("无法正确降维！")
  else:
    print(Y)
    plt.plot(Y[0:49,1],Y[0:49,0], '^', markersize=7, color='red', alpha=0.5, label='Virginia')
    plt.plot(Y[50:99,1],Y[50:99,0], 'o', markersize=7, color='blue', alpha=0.5, label='Versicolor')
    plt.plot(Y[100:149,1],Y[100:149,0], 's', markersize=7, color='green', alpha=0.5, label='Setosa')

    plt.xlim([-1.5,1.5])
    plt.ylim([-4,4])

    plt.xlabel('2 principal component')
    plt.ylabel('1 principal component')
    plt.legend()
    plt.title('Iris ReduceDimension images')

    plt.tight_layout
    plt.grid()
    plt.show()
    