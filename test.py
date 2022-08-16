import numpy as np
from matplotlib import pyplot as plt


# 生成3D示例数据

mu_vec1 = np.array([0,0,0]) # 均值向量
cov_mat1 = np.array([[1,0,0],[0,1,0],[0,0,1]]) # 协方差矩阵

class1_sample = np.random.multivariate_normal(mu_vec1, cov_mat1, 20)
class2_sample = np.random.multivariate_normal(mu_vec1 + 1, cov_mat1, 20)
class3_sample = np.random.multivariate_normal(mu_vec1 + 2, cov_mat1, 20)


# class1_sample.shape -> (20, 3), 20 rows, 3 columns


fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(class1_sample[:,0], class1_sample[:,1], class1_sample[:,2],
           marker='x', color='blue', s=40, label='class 1')
ax.scatter(class2_sample[:,0], class2_sample[:,1], class2_sample[:,2],
           marker='o', color='green', s=40, label='class 2')
ax.scatter(class3_sample[:,0], class3_sample[:,1], class3_sample[:,2],
           marker='^', color='red', s=40, label='class 3')

ax.set_xlabel('variable X')
ax.set_ylabel('variable Y')
ax.set_zlabel('variable Z')

plt.title('3D Scatter Plot')

plt.show()