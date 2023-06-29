import random

import numpy as np
import htucker as ht

num_dim = random.randint(4,7)
num_dim = 6
print(num_dim)
size = [random.randint(2,10) for _ in range(num_dim)]
print(size)
leaf_ranks = [random.randint(2,max_rank) for max_rank in size]

print(leaf_ranks)
num_transfer_nodes = num_dim-2
transfer_ranks = [random.randint(min(leaf_ranks),max(leaf_ranks)) for _ in range(num_transfer_nodes)]
# transfer_ranks = [random.randint(2,5) for _ in range(num_transfer_nodes)]
# print("Initial transfer ranks:")
# print(transfer_ranks)

if len(transfer_ranks)<6:
    transfer_ranks.extend(leaf_ranks[len(transfer_ranks)-6:])
# print("Transfer ranks after completing to a minimum of 6:")
print(transfer_ranks)
leafs = [np.random.randn(r, n) for r,n in zip(leaf_ranks, size)]
# l_subtree_transfer_ranks, r_subtree_transfer_ranks = ht.split_dimensions(transfer_ranks)
# print(l_subtree_transfer_ranks)
# print(r_subtree_transfer_ranks)
# l_root_rank=l_subtree_transfer_ranks.pop(0)
# r_root_rank=r_subtree_transfer_ranks.pop(0)

l_root_rank=transfer_ranks.pop(0)
r_root_rank=transfer_ranks.pop(0)
# print(l_root_rank,r_root_rank)

root=ht.TuckerCore(core=np.random.randn(l_root_rank,r_root_rank),dims=size)

l_dims,r_dims=ht.split_dimensions(size)
# print(l_dims,r_dims)

l_rank=transfer_ranks.pop(0)
r_rank=transfer_ranks.pop(0)
# print(l_rank,r_rank)
root.left=ht.TuckerCore(core=np.random.randn(l_rank,r_rank,l_root_rank),parent=root,dims=l_dims)

l_rank=transfer_ranks.pop(0)
r_rank=transfer_ranks.pop(0)
# print(l_rank,r_rank)
root.right=ht.TuckerCore(core=np.random.randn(l_rank,r_rank,r_root_rank),parent=root,dims=r_dims)

# root.right=None
completed_cores = []
completed_cores.append(root)
completed_cores.append(root.left)
completed_cores.append(root.right)

transfer_cores = []
transfer_cores.append(root.left)
transfer_cores.append(root.right)

while transfer_cores:
    node=transfer_cores.pop(-1)
    # print("node dims:",node.dims)
    node.get_ranks()
    # print("node ranks:",node.ranks)
    l_dims,r_dims=ht.split_dimensions(node.dims)
    # print(l_dims,r_dims)
    # print("Remaining leaf ranks:",leaf_ranks)
    # print("Remaining transfer ranks:",transfer_ranks)

    # print("Right dimensions:",r_dims)
    if len(r_dims)==1:
        # leaf_rank=leaf_ranks.pop(-1)
        leaf_rank=node.ranks[1]
        leaf_dim=r_dims[0]
        print(leaf_rank,leaf_dim)
        # print(leaf_rank,leaf_dim)
        node.right=ht.TuckerLeaf(matrix=np.random.randn(leaf_dim,leaf_rank),parent=node,dims=r_dims)
    elif len(r_dims)==2:
        r_rank=leaf_ranks.pop(-1)
        l_rank=leaf_ranks.pop(-1)
        node.right=ht.TuckerCore(core=np.random.randn(l_rank,r_rank,node.ranks[1]),parent=node,dims=r_dims)
        transfer_cores.insert(0,node.right)
        completed_cores.append(node.right)
        # node.right=ht.TuckerLeaf()
        # node.left=ht.TuckerLeaf()
    elif len(r_dims)==3:
        r_rank=leaf_ranks.pop(-1)
        l_rank=transfer_ranks.pop(0)
        node.right=ht.TuckerCore(core=np.random.randn(l_rank,r_rank,node.ranks[1]),parent=node,dims=r_dims)
        transfer_cores.insert(0,node.right)
        completed_cores.append(node.right)
    else:
        l_rank=transfer_ranks.pop(0)
        r_rank=transfer_ranks.pop(0)
        node.right=ht.TuckerCore(core=np.random.randn(l_rank,r_rank,node.ranks[1]),parent=node,dims=r_dims)
        transfer_cores.insert(0,node.right)
        completed_cores.append(node.right)

    # print("Left dimensions:",l_dims)
    if len(l_dims)==1:
        # leaf_rank=leaf_ranks.pop(-1)
        leaf_rank=node.ranks[0]
        leaf_dim=l_dims[0]
        print(leaf_rank,leaf_dim)
        node.left=ht.TuckerLeaf(matrix=np.random.randn(leaf_dim,leaf_rank),parent=node,dims=l_dims)
    elif len(l_dims)==2:
        r_rank=leaf_ranks.pop(-1)
        l_rank=leaf_ranks.pop(-1)
        node.left=ht.TuckerCore(core=np.random.randn(l_rank,r_rank,node.ranks[0]),parent=node,dims=l_dims)
        transfer_cores.insert(0,node.left)
        completed_cores.append(node.left)
    elif len(l_dims)==3:
        r_rank=leaf_ranks.pop(-1)
        l_rank=transfer_ranks.pop(0)
        node.left=ht.TuckerCore(core=np.random.randn(l_rank,r_rank,node.ranks[0]),parent=node,dims=l_dims)
        transfer_cores.insert(0,node.left)
        completed_cores.append(node.left)
    else:
        l_rank=transfer_ranks.pop(0)
        r_rank=transfer_ranks.pop(0)
        node.left=ht.TuckerCore(core=np.random.randn(l_rank,r_rank,node.ranks[0]),parent=node,dims=l_dims)
        transfer_cores.insert(0,node.left)
        completed_cores.append(node.left)

transfer_cores=completed_cores.copy()

while completed_cores:
    node=completed_cores.pop(-1)
    node.contract_children()

tens=ht.HTucker()
tens.initialize(node.core)
anan=ht.createDimensionTree(tens.original_shape,2,1)
anan.get_items_from_level()
tens.compress_leaf2root(node.core,dimension_tree=anan)

tens2=ht.HTucker()
tens2.initialize(node.core)
tens2.compress_root2leaf(node.core)

tens.reconstruct()
tens2.reconstruct()

for leaf in tens.leaves:
    print(leaf.core.T.shape)
left=True
right=False
# while l_subtree_transfer_ranks:

#     transfer_rank=l_subtree_transfer_ranks.pop(0)
#     if left:
#         left = False
#         right = True


#     else:
#         right = False
#         left = True

# left=True
# right=False
print(np.allclose((tens.root.core-node.core),np.zeros_like(node.core),atol=5e-8))
print(np.allclose((tens2.root.core-node.core),np.zeros_like(node.core),atol=5e-8))
transfer_tensors = []
