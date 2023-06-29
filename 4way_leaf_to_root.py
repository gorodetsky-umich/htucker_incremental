import random

import numpy as np
import htucker as ht

# Set seed
seed = 2
np.random.seed(seed)

# Create 4-way random tensor
num_dim = 4
dimensions = [11, 4, 7, 5]

leaf_ranks = [3, 2, 5, 4]

leafs = [np.random.randn(r, n) for r,n in zip(leaf_ranks, dimensions)]

transfer_ranks = [3, 6]

# Create leaves
transfer_tensors = [
np.random.randn(leaf_ranks[0], leaf_ranks[1], transfer_ranks[0]),
np.random.randn(leaf_ranks[2], leaf_ranks[3], transfer_ranks[1])
]

# Create root node
root = np.random.randn(transfer_ranks[0], transfer_ranks[1])

# Create transfer nodes
eval_left = np.einsum('ij,kl,ikr->jlr', leafs[0], leafs[1], transfer_tensors[0])
eval_right = np.einsum('ij,kl,ikr->jlr', leafs[2], leafs[3], transfer_tensors[1])

# Create tensor
tensor = np.einsum('ijk,lmn,kn->ijlm',eval_left, eval_right, root)

core, [leaf1, leaf2, leaf3, leaf4]=ht.hosvd(tensor)


2+2