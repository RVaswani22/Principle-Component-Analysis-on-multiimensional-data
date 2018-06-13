import numpy as np
import io
from math import sqrt
from numpy import linalg
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.decomposition import TruncatedSVD
from sklearn import decomposition
#----------------To get number of columns-------------------------------
first_line = []
filename='pca_demo.txt'
with open(filename, 'r') as f:
    first_line = f.readline()
n = len(first_line.split("	"))
#-------------------To read as a matrix---------------------------------
input_file = np.loadtxt(filename,usecols=range(0,n-1))
f=open(filename,"r")
lines=f.readlines()
names=[]
for x in lines:
    names.append(x.split('\t')[n-1][:-1])
f.close()
#----------------------Mean vector--------------------------------------
X_mean = np.mean(input_file, axis=0)
#----------------------Adjusted matrix----------------------------------
x_adjust = input_file - X_mean
#------------------------Covariance matrix-----------------------------
x_covar = np.cov(np.transpose(x_adjust))
#----------------------Eigen Values and Eigenvectors-------------------
eig_vals, eig_vecs = np.linalg.eig(x_covar)
sortedvals=np.argsort(eig_vals)
ind = np.sort(np.argpartition(eig_vals, -2)[-2:])
#------------------------------Finding final values-------------------
yi = np.dot(x_adjust, eig_vecs[:,(ind[0],ind[1])])
#----------Plotting-------------
output=list(set(names))
xcoord=yi[:,(0)]
ycoord=yi[:,(1)]
for i in range(len(output)):
    indices = [j for j, d in enumerate(names) if output[i] == names[j]]
    plt.scatter(xcoord[indices],ycoord[indices],cmap=cm.get_cmap('Dark2'),label=output[i])
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title("PCA on "+filename)
plt.show()
#----------------SVD-----
#U, s, V = np.linalg.svd(input_file, full_matrices=False)
#svdout=np.dot(U,np.diag(s))
svd = TruncatedSVD(n_components=2)
svdout=svd.fit_transform(input_file)
xcoord=svdout[:,(0)]
ycoord=svdout[:,(1)]
for i in range(len(output)):
    indices = [j for j, d in enumerate(names) if output[i] == names[j]]
    plt.scatter(xcoord[indices],ycoord[indices],cmap=cm.get_cmap('Dark2'),label=output[i])
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title("SVD on "+filename)
plt.show()
#----TSNE-----------------
tsne = TSNE(n_components=2)
tsneout=tsne.fit_transform(input_file)
xcoord=tsneout[:,(0)]
ycoord=tsneout[:,(1)]
for i in range(len(output)):
    indices = [j for j, d in enumerate(names) if output[i] == names[j]]
    plt.scatter(xcoord[indices],ycoord[indices],cmap=cm.get_cmap('Dark2'),label=output[i])
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title("T-SNE on "+filename)
plt.show()