import unittest
import numpy as np
import sys 
sys.path.append('./')
import htucker as ht
# from ..algs import HTucker 

seed = 2
np.random.seed(seed)
        
class TestCase(unittest.TestCase):

    def setUp(self):

        num_dim = 4
        self.size = [3, 4, 7, 5]

        leaf_ranks = [3, 2, 5, 4]
        
        leafs = [np.random.randn(r, n) for r,n in zip(leaf_ranks, self.size)]

        transfer_ranks = [3, 6]

        transfer_tensors = [
            np.random.randn(leaf_ranks[0], leaf_ranks[1], transfer_ranks[0]),
            np.random.randn(leaf_ranks[2], leaf_ranks[3], transfer_ranks[1])
        ]

        root = np.random.randn(transfer_ranks[0], transfer_ranks[1])
        
        eval_left = np.einsum('ij,kl,ikr->jlr', leafs[0], leafs[1], transfer_tensors[0])
        eval_right = np.einsum('ij,kl,ikr->jlr', leafs[2], leafs[3], transfer_tensors[1])

        self.tensor = np.einsum('ijk,lmn,kn->ijlm',eval_left, eval_right, root)
        self.koldaTensor=np.array(
            [
                [
                    [1,13],
                    [4,16],
                    [7,19],
                    [10,22]
                    ],
                [
                    [2,14],
                    [5,17],
                    [8,20],
                    [11,23]
                    ],
                [
                    [3,15],
                    [6,18],
                    [9,21],
                    [12,24]
                    ]
                ]
                
        )
        

    def test_add_edge(self):

        # print("\n", self.tensor)
        self.assertEqual(self.size[0], self.tensor.shape[0])
        self.assertEqual(self.size[1], self.tensor.shape[1])
        self.assertEqual(self.size[2], self.tensor.shape[2])
        self.assertEqual(self.size[3], self.tensor.shape[3])

        # print("\n",self.koldaTensor.shape)
        # # This is the way that one should compute the mode-n reshaping in the sense of Kolda!
        # print(self.koldaTensor.reshape(3,8,order='F'))
        # print(self.koldaTensor.transpose(1,0,2).reshape(4,6,order='F'))
        # print(self.koldaTensor.transpose(2,0,1).reshape(2,12,order='F'))
        
        # assert all false
        self.assertTrue(True)
    
    def test_hosvd(self):
        core,matrices=ht.algs.hosvd(self.tensor)
        reconstruction=np.einsum('ij,kl,mn,op,jlnp->ikmo',matrices[0],matrices[1],matrices[2],matrices[3],core)
        self.assertTrue(np.allclose((reconstruction-self.tensor),np.zeros_like(reconstruction)))


        
        
        
if __name__ == '__main__':
    unittest.main()
