import random

import numpy as np
import htucker as ht
# import DaMAT as dmt

from PIL import Image
# from DaMAT.utils import primes

SEED = 1905
np.random.seed(SEED)

# random_tensor = np.random.randn(110,170,90)
random_tensor = np.array(Image.open("31.png"))
random_tensor = random_tensor.reshape(14,15,16,10,3,order="F")
# random_tensor = random_tensor.reshape(6,5,7,8,10,6,order="F")
# random_tensor = random_tensor.reshape(2,3,5,7,2,2,2,2,5,2,3,order="F")
similar_tensor = np.array(Image.open("32.png")).reshape(random_tensor.shape,order="F")
different_tensor = np.array(Image.open("322.png")).reshape(random_tensor.shape,order="F")

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

if False:
    # if len(random_tensor.shape)==5:
    #     ## Project similar tensor, this is hard-coded and only works for dim=5
    #     anan=similar_tensor.copy()
    #     anan=np.einsum(
    #         "abcde,af,bg->fgcde",
    #         anan,tens._dimension_tree._level_items[-1][0].real_node.core,tens._dimension_tree._level_items[-1][1].real_node.core,
    #         optimize=True,order="F"
    #     )
    #     anan=np.einsum(
    #         "abcde,abf,cg,dh,ei->fghi",
    #         anan,tens._dimension_tree._level_items[-2][0].real_node.core,tens._dimension_tree._level_items[-2][1].real_node.core,tens._dimension_tree._level_items[-2][2].real_node.core,tens._dimension_tree._level_items[-2][3].real_node.core,
    #         optimize=True,order="F"
    #     )
    #     anan=np.einsum(
    #         "abcd,abe,cdf->ef",
    #         anan,tens._dimension_tree._level_items[-3][0].real_node.core,tens._dimension_tree._level_items[-3][1].real_node.core,
    #         optimize=True,order="F"
    #     )
    #     similar_core=anan.copy()
    #     deneme_core = tens.project(similar_tensor)
    #     anan=np.einsum(
    #         "ef,abe,cdf->abcd",
    #         anan,tens._dimension_tree._level_items[-3][0].real_node.core,tens._dimension_tree._level_items[-3][1].real_node.core,
    #         optimize=True,order="F"
    #     )
    #     anan=np.einsum(
    #         "fghi,abf,cg,dh,ei->abcde",
    #         anan,tens._dimension_tree._level_items[-2][0].real_node.core,tens._dimension_tree._level_items[-2][1].real_node.core,tens._dimension_tree._level_items[-2][2].real_node.core,tens._dimension_tree._level_items[-2][3].real_node.core,
    #         optimize=True,order="F"
    #     )
    #     anan=np.einsum(
    #         "fgcde,af,bg->abcde",
    #         anan,tens._dimension_tree._level_items[-1][0].real_node.core,tens._dimension_tree._level_items[-1][1].real_node.core,
    #         optimize=True,order="F"
    #     )    
    #     similar_reconstructed=anan.copy()
    #     anan = tens.reconstruct(deneme_core)
    #     ## Project different tensor, this is hard-coded and only works for dim=5
    #     anan=different_tensor.copy()
    #     anan=np.einsum(
    #         "abcde,af,bg->fgcde",
    #         anan,tens._dimension_tree._level_items[-1][0].real_node.core,tens._dimension_tree._level_items[-1][1].real_node.core,
    #         optimize=True,order="F"
    #     )
    #     anan=np.einsum(
    #         "abcde,abf,cg,dh,ei->fghi",
    #         anan,tens._dimension_tree._level_items[-2][0].real_node.core,tens._dimension_tree._level_items[-2][1].real_node.core,tens._dimension_tree._level_items[-2][2].real_node.core,tens._dimension_tree._level_items[-2][3].real_node.core,
    #         optimize=True,order="F"
    #     )
    #     anan=np.einsum(
    #         "abcd,abe,cdf->ef",
    #         anan,tens._dimension_tree._level_items[-3][0].real_node.core,tens._dimension_tree._level_items[-3][1].real_node.core,
    #         optimize=True,order="F"
    #     )
    #     different_core=anan.copy()
    #     anan=np.einsum(
    #         "ef,abe,cdf->abcd",
    #         anan,tens._dimension_tree._level_items[-3][0].real_node.core,tens._dimension_tree._level_items[-3][1].real_node.core,
    #         optimize=True,order="F"
    #     )
    #     anan=np.einsum(
    #         "fghi,abf,cg,dh,ei->abcde",
    #         anan,tens._dimension_tree._level_items[-2][0].real_node.core,tens._dimension_tree._level_items[-2][1].real_node.core,tens._dimension_tree._level_items[-2][2].real_node.core,tens._dimension_tree._level_items[-2][3].real_node.core,
    #         optimize=True,order="F"
    #     )
    #     anan=np.einsum(
    #         "fgcde,af,bg->abcde",
    #         anan,tens._dimension_tree._level_items[-1][0].real_node.core,tens._dimension_tree._level_items[-1][1].real_node.core,
    #         optimize=True,order="F"
    #     )    
    #     different_reconstructed=anan.copy()
    2+2

similar_core = tens.project(similar_tensor)
similar_reconstructed = tens.reconstruct(similar_core)
different_core = tens.project(different_tensor)
different_reconstructed = tens.reconstruct(different_core)


allowed_error=np.linalg.norm(different_tensor)*epsilon/np.sqrt(2*len(different_tensor.shape)-3)
anan=different_tensor.copy()

# Update first leaf
anan_tempProj=np.einsum(
    "abcde,af,gf->gbcde",anan,tens._dimension_tree._level_items[-1][0].real_node.core,tens._dimension_tree._level_items[-1][0].real_node.core,
    optimize=True,order="F"
)
anan_residual=anan-anan_tempProj
u,s,_ = np.linalg.svd(ht.mode_n_unfolding(anan_residual,0),full_matrices=False)
u=u[:,np.cumsum((s**2)[::-1])[::-1]>(allowed_error)**2]
s=s[np.cumsum((s**2)[::-1])[::-1]>(allowed_error)**2]
tens._dimension_tree._level_items[-1][0].real_node.core= np.concatenate((tens._dimension_tree._level_items[-1][0].real_node.core,u),axis=1)
tens._dimension_tree._level_items[-1][0].real_node.rank+=u.shape[1]
# TODO: Need to implement rank matching


# Update second leaf
anan_tempProj=np.einsum(
    "abcde,bf,gf->agcde",anan,tens._dimension_tree._level_items[-1][1].real_node.core,tens._dimension_tree._level_items[-1][1].real_node.core,
    optimize=True,order="F"
)
anan_residual=anan-anan_tempProj
u,s,_ = np.linalg.svd(ht.mode_n_unfolding(anan_residual,1),full_matrices=False)
u=u[:,np.cumsum((s**2)[::-1])[::-1]>(allowed_error)**2]
s=s[np.cumsum((s**2)[::-1])[::-1]>(allowed_error)**2]
tens._dimension_tree._level_items[-1][1].real_node.core= np.concatenate((tens._dimension_tree._level_items[-1][1].real_node.core,u),axis=1)
tens._dimension_tree._level_items[-1][1].real_node.rank+=u.shape[1]
# TODO: Need to implement rank matching

# Project new tensor through last layer
anan=np.einsum(
        "abcde,af,bg->fgcde",
        anan,tens._dimension_tree._level_items[-1][0].real_node.core,tens._dimension_tree._level_items[-1][1].real_node.core,
        optimize=True,order="F"
    )

# Update first transfer node
anan_tempProj=np.einsum(
    "abcde,abf,ghf->ghcde",anan,tens._dimension_tree._level_items[-2][0].real_node.core,tens._dimension_tree._level_items[-2][0].real_node.core,
    optimize=True,order="F"
)
anan_residual=anan-anan_tempProj
u,s,_ = np.linalg.svd(ht.mode_n_unfolding(anan_residual.reshape(
    [np.prod(tens._dimension_tree._level_items[-2][0].real_node.shape[:-1])]+list(anan_tempProj.shape[2:]),order="F"),0
    ),full_matrices=False)
u=u[:,np.cumsum((s**2)[::-1])[::-1]>(allowed_error)**2]
s=s[np.cumsum((s**2)[::-1])[::-1]>(allowed_error)**2]
tens._dimension_tree._level_items[-2][0].real_node.core = np.concatenate((tens._dimension_tree._level_items[-2][0].real_node.core,u.reshape(13,14,-1,order="F")),axis=-1)
tens._dimension_tree._level_items[-2][0].real_node.get_ranks()

# RANK MATCHING
if tens._dimension_tree._level_items[-2][0].parent._dimension_index.index(tens._dimension_tree._level_items[-2][0]._dimension_index[0]) == 0:
    ranks=tens._dimension_tree._level_items[-2][0].real_parent.ranks
    tens._dimension_tree._level_items[-2][0].real_parent.core = np.concatenate((tens._dimension_tree._level_items[-2][0].real_parent.core,np.zeros([u.shape[1]]+ranks[1:])),axis=0)
else:
    ranks=tens._dimension_tree._level_items[-2][0].real_parent.ranks
    tens._dimension_tree._level_items[-2][0].real_parent.core = np.concatenate((tens._dimension_tree._level_items[-2][0].real_parent.core,np.zeros([ranks[0],u.shape[1],ranks[-1]])),axis=1)
tens._dimension_tree._level_items[-2][0].real_parent.get_ranks()


# Update third leaf
anan_tempProj=np.einsum(
    "abcde,cf,gf->abgde",anan,tens._dimension_tree._level_items[-2][1].real_node.core,tens._dimension_tree._level_items[-2][1].real_node.core,
    optimize=True,order="F"
)
anan_residual=anan-anan_tempProj
u,s,_ = np.linalg.svd(ht.mode_n_unfolding(anan_residual,2),full_matrices=False)
u=u[:,np.cumsum((s**2)[::-1])[::-1]>(allowed_error)**2]
s=s[np.cumsum((s**2)[::-1])[::-1]>(allowed_error)**2]
tens._dimension_tree._level_items[-2][1].real_node.core= np.concatenate((tens._dimension_tree._level_items[-2][1].real_node.core,u),axis=1)
tens._dimension_tree._level_items[-2][1].real_node.rank+=u.shape[1]
# Rank matching
if tens._dimension_tree._level_items[-2][1].parent._dimension_index.index(tens._dimension_tree._level_items[-2][1]._dimension_index[0]) == 0:
    ranks=tens._dimension_tree._level_items[-2][1].real_parent.ranks
    tens._dimension_tree._level_items[-2][1].real_parent.core = np.concatenate((tens._dimension_tree._level_items[-2][1].real_parent.core,np.zeros([u.shape[1]]+ranks[1:])),axis=0)
else:
    ranks=tens._dimension_tree._level_items[-2][1].real_parent.ranks
    tens._dimension_tree._level_items[-2][1].real_parent.core = np.concatenate((tens._dimension_tree._level_items[-2][1].real_parent.core,np.zeros([ranks[0],u.shape[1],ranks[-1]])),axis=1)
tens._dimension_tree._level_items[-2][1].real_parent.get_ranks()


# Update fourth leaf
anan_tempProj=np.einsum(
    "abcde,df,gf->abcge",anan,tens._dimension_tree._level_items[-2][2].real_node.core,tens._dimension_tree._level_items[-2][2].real_node.core,
    optimize=True,order="F"
)
anan_residual=anan-anan_tempProj
u,s,_ = np.linalg.svd(ht.mode_n_unfolding(anan_residual,3),full_matrices=False)
u=u[:,np.cumsum((s**2)[::-1])[::-1]>(allowed_error)**2]
s=s[np.cumsum((s**2)[::-1])[::-1]>(allowed_error)**2]
tens._dimension_tree._level_items[-2][2].real_node.core= np.concatenate((tens._dimension_tree._level_items[-2][2].real_node.core,u),axis=1)
tens._dimension_tree._level_items[-2][2].real_node.rank+=u.shape[1]
# Rank matching
if tens._dimension_tree._level_items[-2][2].parent._dimension_index.index(tens._dimension_tree._level_items[-2][2]._dimension_index[0]) == 0:
    ranks=tens._dimension_tree._level_items[-2][2].real_parent.ranks
    tens._dimension_tree._level_items[-2][2].real_parent.core = np.concatenate((tens._dimension_tree._level_items[-2][2].real_parent.core,np.zeros([u.shape[1]]+ranks[1:])),axis=0)
else:
    ranks=tens._dimension_tree._level_items[-2][2].real_parent.ranks
    tens._dimension_tree._level_items[-2][2].real_parent.core = np.concatenate((tens._dimension_tree._level_items[-2][2].real_parent.core,np.zeros([ranks[0],u.shape[1],ranks[-1]])),axis=1)
tens._dimension_tree._level_items[-2][2].real_parent.get_ranks()

# Update fifth leaf
anan_tempProj=np.einsum(
    "abcde,ef,gf->abcdg",anan,tens._dimension_tree._level_items[-2][3].real_node.core,tens._dimension_tree._level_items[-2][3].real_node.core,
    optimize=True,order="F"
)
anan_residual=anan-anan_tempProj
u,s,_ = np.linalg.svd(ht.mode_n_unfolding(anan_residual,4),full_matrices=False)
u=u[:,np.cumsum((s**2)[::-1])[::-1]>(allowed_error)**2]
s=s[np.cumsum((s**2)[::-1])[::-1]>(allowed_error)**2]
tens._dimension_tree._level_items[-2][3].real_node.core= np.concatenate((tens._dimension_tree._level_items[-2][3].real_node.core,u),axis=1)
tens._dimension_tree._level_items[-2][3].real_node.rank+=u.shape[1]
# Rank matching
if tens._dimension_tree._level_items[-2][3].parent._dimension_index.index(tens._dimension_tree._level_items[-2][3]._dimension_index[0]) == 0:
    ranks=tens._dimension_tree._level_items[-2][3].real_parent.ranks
    tens._dimension_tree._level_items[-2][3].real_parent.core = np.concatenate((tens._dimension_tree._level_items[-2][3].real_parent.core,np.zeros([u.shape[1]]+ranks[1:])),axis=0)
else:
    ranks=tens._dimension_tree._level_items[-2][3].real_parent.ranks
    tens._dimension_tree._level_items[-2][3].real_parent.core = np.concatenate((tens._dimension_tree._level_items[-2][3].real_parent.core,np.zeros([ranks[0],u.shape[1],ranks[-1]])),axis=1)
tens._dimension_tree._level_items[-2][3].real_parent.get_ranks()

# Project new tensor through last layer
anan=np.einsum(
        "abcde,abf,cg,dh,ei->fghi",
        anan,tens._dimension_tree._level_items[-2][0].real_node.core,tens._dimension_tree._level_items[-2][1].real_node.core,tens._dimension_tree._level_items[-2][2].real_node.core,tens._dimension_tree._level_items[-2][3].real_node.core,
        optimize=True,order="F"
    )

# Update first transfer node
anan_tempProj=np.einsum(
    "abcd,abf,ghf->ghcd",anan,tens._dimension_tree._level_items[-3][0].real_node.core,tens._dimension_tree._level_items[-3][0].real_node.core,
    optimize=True,order="F"
)
anan_residual=anan-anan_tempProj
u,s,_ = np.linalg.svd(ht.mode_n_unfolding(anan_residual.reshape(
    [np.prod(tens._dimension_tree._level_items[-3][0].real_node.shape[:-1])]+list(anan_tempProj.shape[2:]),order="F"),0
    ),full_matrices=False)
u=u[:,np.cumsum((s**2)[::-1])[::-1]>(allowed_error)**2]
s=s[np.cumsum((s**2)[::-1])[::-1]>(allowed_error)**2]
tens._dimension_tree._level_items[-3][0].real_node.core = np.concatenate((tens._dimension_tree._level_items[-3][0].real_node.core,u.reshape(41,11,-1,order="F")),axis=-1)
tens._dimension_tree._level_items[-3][0].real_node.get_ranks()
if tens._dimension_tree._level_items[-3][0].parent._dimension_index.index(tens._dimension_tree._level_items[-3][0]._dimension_index[0]) == 0:
    ranks=tens._dimension_tree._level_items[-3][0].real_parent.ranks
    tens._dimension_tree._level_items[-3][0].real_parent.core = np.concatenate((tens._dimension_tree._level_items[-3][0].real_parent.core,np.zeros([u.shape[1]]+ranks[1:])),axis=0)
else:
    ranks=tens._dimension_tree._level_items[-3][0].real_parent.ranks
    tens._dimension_tree._level_items[-3][0].real_parent.core = np.concatenate((tens._dimension_tree._level_items[-3][0].real_parent.core,np.zeros([ranks[0]]+[u.shape[1]]+ranks[2:])),axis=1)
tens._dimension_tree._level_items[-3][0].real_parent.get_ranks()


# Update second transfer node
anan_tempProj=np.einsum(
    "abcd,cdf,ghf->abgh",anan,tens._dimension_tree._level_items[-3][1].real_node.core,tens._dimension_tree._level_items[-3][1].real_node.core,
    optimize=True,order="F"
)
anan_residual=anan-anan_tempProj
u,s,_ = np.linalg.svd(ht.mode_n_unfolding(anan_residual.reshape(
    list(anan_tempProj.shape[:2])+[np.prod(tens._dimension_tree._level_items[-3][1].real_node.shape[:-1])],order="F"),2
    ),full_matrices=False)
u=u[:,np.cumsum((s**2)[::-1])[::-1]>(allowed_error)**2]
s=s[np.cumsum((s**2)[::-1])[::-1]>(allowed_error)**2]
tens._dimension_tree._level_items[-3][1].real_node.core = np.concatenate((tens._dimension_tree._level_items[-3][1].real_node.core,u.reshape(10,3,-1,order="F")),axis=-1)
tens._dimension_tree._level_items[-3][1].real_node.get_ranks()

if tens._dimension_tree._level_items[-3][1].parent._dimension_index.index(tens._dimension_tree._level_items[-3][1]._dimension_index[0]) == 0:
    ranks=tens._dimension_tree._level_items[-3][1].real_parent.ranks
    tens._dimension_tree._level_items[-3][1].real_parent.core = np.concatenate((tens._dimension_tree._level_items[-3][1].real_parent.core,np.zeros([u.shape[1]]+ranks[1:])),axis=0)
else:
    ranks=tens._dimension_tree._level_items[-3][1].real_parent.ranks
    tens._dimension_tree._level_items[-3][1].real_parent.core = np.concatenate((tens._dimension_tree._level_items[-3][1].real_parent.core,np.zeros([ranks[0]]+[u.shape[1]]+ranks[2:])),axis=1)
tens._dimension_tree._level_items[-3][1].real_parent.get_ranks()


anan=similar_tensor.copy()
anan=np.einsum(
    "abcde,af,bg->fgcde",
    anan,tens._dimension_tree._level_items[-1][0].real_node.core,tens._dimension_tree._level_items[-1][1].real_node.core,
    optimize=True,order="F"
)
anan=np.einsum(
    "abcde,abf,cg,dh,ei->fghi",
    anan,tens._dimension_tree._level_items[-2][0].real_node.core,tens._dimension_tree._level_items[-2][1].real_node.core,tens._dimension_tree._level_items[-2][2].real_node.core,tens._dimension_tree._level_items[-2][3].real_node.core,
    optimize=True,order="F"
)
anan=np.einsum(
    "abcd,abe,cdf->ef",
    anan,tens._dimension_tree._level_items[-3][0].real_node.core,tens._dimension_tree._level_items[-3][1].real_node.core,
    optimize=True,order="F"
)
similar_core2=anan.copy()
anan=np.einsum(
    "ef,abe,cdf->abcd",
    anan,tens._dimension_tree._level_items[-3][0].real_node.core,tens._dimension_tree._level_items[-3][1].real_node.core,
    optimize=True,order="F"
)
anan=np.einsum(
    "fghi,abf,cg,dh,ei->abcde",
    anan,tens._dimension_tree._level_items[-2][0].real_node.core,tens._dimension_tree._level_items[-2][1].real_node.core,tens._dimension_tree._level_items[-2][2].real_node.core,tens._dimension_tree._level_items[-2][3].real_node.core,
    optimize=True,order="F"
)
anan=np.einsum(
    "fgcde,af,bg->abcde",
    anan,tens._dimension_tree._level_items[-1][0].real_node.core,tens._dimension_tree._level_items[-1][1].real_node.core,
    optimize=True,order="F"
)    
similar_reconstructed2=anan.copy()
## Project different tensor, this is hard-coded and only works for dim=5
anan=different_tensor.copy()
anan=np.einsum(
    "abcde,af,bg->fgcde",
    anan,tens._dimension_tree._level_items[-1][0].real_node.core,tens._dimension_tree._level_items[-1][1].real_node.core,
    optimize=True,order="F"
)
anan=np.einsum(
    "abcde,abf,cg,dh,ei->fghi",
    anan,tens._dimension_tree._level_items[-2][0].real_node.core,tens._dimension_tree._level_items[-2][1].real_node.core,tens._dimension_tree._level_items[-2][2].real_node.core,tens._dimension_tree._level_items[-2][3].real_node.core,
    optimize=True,order="F"
)
anan=np.einsum(
    "abcd,abe,cdf->ef",
    anan,tens._dimension_tree._level_items[-3][0].real_node.core,tens._dimension_tree._level_items[-3][1].real_node.core,
    optimize=True,order="F"
)
different_core2=anan.copy()
anan=np.einsum(
    "ef,abe,cdf->abcd",
    anan,tens._dimension_tree._level_items[-3][0].real_node.core,tens._dimension_tree._level_items[-3][1].real_node.core,
    optimize=True,order="F"
)
anan=np.einsum(
    "fghi,abf,cg,dh,ei->abcde",
    anan,tens._dimension_tree._level_items[-2][0].real_node.core,tens._dimension_tree._level_items[-2][1].real_node.core,tens._dimension_tree._level_items[-2][2].real_node.core,tens._dimension_tree._level_items[-2][3].real_node.core,
    optimize=True,order="F"
)
anan=np.einsum(
    "fgcde,af,bg->abcde",
    anan,tens._dimension_tree._level_items[-1][0].real_node.core,tens._dimension_tree._level_items[-1][1].real_node.core,
    optimize=True,order="F"
)    
different_reconstructed2=anan.copy()



np.linalg.norm(different_tensor-different_reconstructed)/np.linalg.norm(different_tensor)
np.linalg.norm(different_tensor-different_reconstructed2)/np.linalg.norm(different_tensor)
np.linalg.norm(similar_tensor-similar_reconstructed)/np.linalg.norm(similar_tensor)
np.linalg.norm(similar_tensor-similar_reconstructed2)/np.linalg.norm(similar_tensor)

tens.reconstruct()
print(np.linalg.norm(tens.root.core-random_tensor)/tensor_norm)

# print()
# print("TT")
# tens=dmt.ttObject(random_tensor[...,None],epsilon=epsilon)
# tens.ttDecomp()
# print(tens.ttRanks)
# print(tens.compressionRatio)
# print(np.linalg.norm(tens.reconstruct(tens.ttCores[-1]).squeeze()-random_tensor)/tensor_norm)

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
