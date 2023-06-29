import time
import numpy as np
import DaMAT as dmt

np.set_printoptions(suppress=False,linewidth=np.nan)
def minor_mask(A, i, j):
    """Own solution, based on NumPy fancy indexing"""
    mask = np.ones_like(A, dtype=bool)
    mask[i, :] = False
    mask[:, j] = False

    minor = A[mask].reshape(A.shape[0] - 1, A.shape[1] - 1)

    del mask

    return minor

size=100

a=np.random.randn(size,size)
a=a@a.T

while np.any(np.linalg.eigvalsh(a)<=0):
    print("non positive definite a matrix, generating another one")
    a=np.random.rand(size,size)
    a=a@a.T
sigma = np.diag(np.random.rand(size))

# minors = np.zeros_like(a)
determinants = np.zeros_like(a)
matrix=sigma@a@sigma
# for ii in range(a.shape[0]):
#     for jj in range(a.shape[1]):
#         minors[ii,jj]=np.linalg.det(minor_mask(np.eye(size)+matrix,ii,jj))

# print(np.argsort(np.abs(minors).sum(1))[:10])
# print(np.argsort((minors).sum(1))[:10])
# print(np.argsort((matrix).sum(1))[::-1][:10])
print(np.argsort(np.abs(matrix).sum(1))[::-1][:10])
print(np.argsort(np.linalg.norm(matrix,axis=1))[::-1][:10])
# print(np.argsort(np.linalg.norm(minors,axis=1))[::-1][:10])
# print(np.argsort((matrix).sum(0)))
# print(np.argsort(np.abs(matrix).sum(0)))
tic=time.time()
for ii in range(a.shape[0]):
    for jj in range(ii,a.shape[1]):
        # print(ii,jj)
        # mask=np.zeros(a.shape[0])
        # mask[ii],mask[jj]=1,1
        # w=np.diag(mask)
        # determinants[ii,jj]=np.linalg.det(np.eye(a.shape[0])+w.T@w@matrix@w.T@w)
        # determinants[ii,jj]=np.linalg.det(np.eye(a.shape[0])+w@matrix@w.T)
        mask=np.zeros(a.shape[0],dtype=bool)
        mask[np.array([ii,jj])]=1
        determinants[ii,jj]=(np.linalg.det(matrix[mask,:][:,mask]+np.eye(mask.sum())))
        
print()
print(round(time.time()-tic,4))
# print(np.round(minors,3))
# print(np.round(determinants,0))
argsorts=np.argsort(determinants,axis=None)[::-1]
# print(argsorts[:10])
print()
print(argsorts[:10]//size)
print(argsorts[:10]%size)
# print(determinants.reshape(-1)[np.array(argsorts[:10])])
print(np.round(determinants[argsorts[:10]//size,argsorts[:10]%size],3))
print(np.round(np.amax(determinants),4))

catgelLocation = "/home/doruk/bayesianOED/catgelData/3rd/1/"
data=np.load(catgelLocation+"catgelData_1905.npy")[...,None]

dataSet= dmt.ttObject(data=data,epsilon=0.1)
dataSet.ttDecomp()
data=np.load(catgelLocation+"catgelData_1923.npy")[...,None]
dataSet.ttICE(data,epsilon=0.1)
data=np.load(catgelLocation+"catgelData_1881.npy")[...,None]
dataSet.ttICE(data,epsilon=0.1)
data=np.load(catgelLocation+"catgelData_2023.npy")[...,None]
dataSet.ttICE(data,epsilon=0.1)
data=np.load(catgelLocation+"catgelData_3131.npy")[...,None]
dataSet.ttICE(data,epsilon=0.1)
data=np.load(catgelLocation+"catgelData_0.npy")[...,None]
dataSet.ttICE(data,epsilon=0.1)
data=np.load(catgelLocation+"catgelData_1.npy")[...,None]
dataSet.ttICE(data,epsilon=0.1)


2+2
