import random

import numpy as np
import htucker as ht
import DaMAT as dmt

from PIL import Image
from DaMAT.utils import primes

SEED = 1905
np.random.seed(SEED)

# random_tensor = np.random.randn(110,170,90)
random_tensor = np.array(Image.open("31.png"))
# random_tensor = random_tensor.reshape(14,15,16,10,3,order="F")
# random_tensor = random_tensor.reshape(6,5,7,8,10,6,order="F")
random_tensor = random_tensor.reshape(2,3,5,7,2,2,2,2,5,2,3,order="F")

# random_tensor = np.load("/home/doruk/bayesianOED/catgelData/3rd/1/catgelData_31.npy")#.transpose(0,1,3,2,4)

tensor_norm = np.linalg.norm(random_tensor)
epsilon=0.1
print(f"Epsilon: {epsilon}")
print(f"Shape: {list(random_tensor.shape)}")
# a,c=ht.hosvd(random_tensor,tol=0,threshold=0)
# # a,b,c=ht.hosvd(random_tensor,tol=0,threshold=0)
# # full_svals, full_lsvs = b.copy(), c.copy()
# anan=a.copy()
# print(anan.shape)
# for lsv in c:
#     # print(anan.shape,lsv.shape)
#     anan=np.tensordot(anan,lsv,axes=[0,-1])
#     # print(anan.shape)
# print(np.linalg.norm(anan-random_tensor)/tensor_norm)
# print(np.prod(random_tensor.shape)/(np.prod(a.shape)+sum([np.prod(kk.shape) for kk in c])))

# a,c=ht.hosvd(random_tensor,tol=epsilon*tensor_norm)
# # a,b,c=ht.hosvd(random_tensor,tol=epsilon*tensor_norm)
# anan=a.copy()
# print(anan.shape)
# for lsv in c:
#     # print(anan.shape,lsv.shape)
#     anan=np.tensordot(anan,lsv,axes=[0,-1])
#     # print(anan.shape)
# print(np.linalg.norm(anan-random_tensor)/tensor_norm)
# print(np.prod(random_tensor.shape)/(np.prod(a.shape)+sum([np.prod(kk.shape) for kk in c])))


# a,c=ht.hosvd(random_tensor,rtol=epsilon,threshold=0)
# # a,b,c=ht.hosvd(random_tensor,rtol=epsilon,threshold=0)
# anan=a.copy()
# print(anan.shape)
# for lsv in c:
#     # print(anan.shape,lsv.shape)
#     anan=np.tensordot(anan,lsv,axes=[0,-1])
#     # print(anan.shape)
# print(np.linalg.norm(anan-random_tensor)/tensor_norm)
# print(np.prod(random_tensor.shape)/(np.prod(a.shape)+sum([np.prod(kk.shape) for kk in c])))

# tensor2 = np.array(Image.open("322.png")).reshape(random_tensor.shape,order="F")
# amcik  = tensor2.copy()
# for lsv in c:
#     amcik=np.tensordot(amcik,lsv@lsv.T,axes=[0,-1])
# print(np.linalg.norm(amcik-tensor2)/np.linalg.norm(tensor2))


print()
print("Htucker")
tens=ht.HTucker()
tens.initialize(random_tensor)
dim_tree=ht.createDimensionTree(tens.original_shape,numSplits=2,minSplitSize=1)
dim_tree.get_items_from_level()
print([len(lv) for lv in dim_tree._level_items])
tens.rtol=epsilon
tens.compress_leaf2root(random_tensor,dimension_tree=dim_tree)
print(tens.compression_ratio)
tens.reconstruct()
print(np.linalg.norm(tens.root.core-random_tensor)/tensor_norm)

print()
print("TT")
tens=dmt.ttObject(random_tensor[...,None],epsilon=epsilon)
tens.ttDecomp()
print(tens.ttRanks)
print(tens.compressionRatio)
print(np.linalg.norm(tens.reconstruct(tens.ttCores[-1]).squeeze()-random_tensor)/tensor_norm)

2+2
# max_allowed_error = tensor_norm*epsilon
# a,b,c=ht.hosvd(random_tensor)
# anan=random_tensor.copy()
# print(anan.shape)
# for svals,lsv in zip(full_svals,full_lsvs):
#     yarrak=((svals**2/max_allowed_error**2)<epsilon)
#     print(yarrak)
#     # print((1-yarrak).sum(),anan.shape,lsv.shape)
#     lsv=lsv[:,:(1-yarrak).sum()]
#     anan=np.tensordot(anan,lsv.T,axes=[0,-1])
#     # print(anan.shape)

# print(anan.shape)
# for svals,lsv in zip(full_svals,full_lsvs):
#     yarrak=((svals**2/max_allowed_error**2)<epsilon)
#     # print((1-yarrak).sum(),anan.shape,lsv.shape)
#     lsv=lsv[:,:(1-yarrak).sum()]
#     anan=np.tensordot(anan,lsv,axes=[0,-1])
#     print(anan.shape)
# print(np.linalg.norm(anan-random_tensor)/tensor_norm)
