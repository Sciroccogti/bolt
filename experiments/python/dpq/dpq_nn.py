'''
@file dpq_nn.py
@author Sciroccogti (scirocco_gti@yeah.net)
@brief 
@date 2022-12-31 12:41:49
@modified: 2022-12-31 23:55:00
'''

import numpy as np
import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"


class DPQNetwork(nn.Module):
    def __init__(self, ncentroids: int, ncodebooks: int, subvect_len: int,
                 centroids: torch.Tensor,
                 tie_in_n_out: bool = True, query_metric: str = "dot", shared_centroids: bool = False,
                 beta: float = 0.0, tau: float = 1.0, softmax_BN: bool = True) -> None:
        '''
        :param ncentroids: K
        :param ncodebooks: C in mithral, or D in kpq
        :param subvect_len: D/C in mithral, or d_in, d_out in kpq
        :param centroids: (ncodebooks, ncentroids, subvect_len)
        :param query_metric: "euclidean" or "dot"
        '''
        if tie_in_n_out != True or shared_centroids or beta != 0.0 or tau != 1.0 or softmax_BN != True:
            raise NotImplementedError(
                "tie_in_n_out, shared_centroids, beta, tau, softmax_BN haven't been implemented yet")
        assert centroids.size() == (ncodebooks, ncentroids, subvect_len)
        assert query_metric in ["dot", "euclidean"], "query_metric only supports dot and euclidean"
        super(DPQNetwork, self).__init__()

        device = "cuda" if torch.cuda.is_available() else "cpu"

        self._ncentroids = ncentroids
        self._ncodebooks = ncodebooks
        self._subvect_len = subvect_len
        self._tie_in_n_out = tie_in_n_out
        self._query_metric = query_metric
        self._shared_centroids = shared_centroids
        self._beta = beta
        self._tau = torch.Tensor([1.0]).to(device)  # constant
        self._softmax_BN = softmax_BN
        self._centroids_k = centroids
        self._centroids_v = self._centroids_k

    def forward(self, inputs: torch.Tensor, is_training: bool = True):
        '''
        :param inputs: (batch_size, ncodebooks, subvect_len)
        '''
        if type(inputs) != torch.Tensor:
            inputs = torch.from_numpy(inputs)
        assert inputs.size(1) == self._ncodebooks and inputs.size(2) == self._subvect_len,\
            f"inputs must be (batch_size, ncodebooks, subvect_len), current is {inputs.size()}"
        if inputs.dtype != torch.double:
            inputs = inputs.double()
        centroids_k = self._centroids_k  # (C, K, subvect_len)
        centroids_v = self._centroids_v

        # TODO: other metrics
        if self._query_metric == "euclidean":
            response = torch.cdist(inputs.transpose(0, 1), centroids_k, 2).transpose(0, 1)
        if self._query_metric == "dot":
            # (C, bs, subvect_len) * (C, subvect_len, K) = (C, bs, K)
            response = torch.matmul(inputs.transpose(0, 1), centroids_k.transpose(1, 2))
            # (C, bs, K) -> (bs, C, K)
            response = response.transpose(0, 1)
        assert(response.size(1) == self._ncodebooks and response.size(2) == self._ncentroids)

        # TODO: softmax_BN

        response_prob = torch.softmax(response / self._tau, dim=-1)

        codes = torch.argmax(response, -1)  # (batch_size, C)

        neighbour_idxs = codes

        if not self._shared_centroids:
            D_base = torch.from_numpy(
                np.array([self._ncentroids*d for d in range(self._ncodebooks)])).to(device)
            neighbour_idxs += D_base
        # centroids_v = centroids_v.reshape((-1, self._subvect_len))# (C, K, subvect_len)
        outputs = torch.index_select(
            centroids_v.reshape((-1, self._subvect_len)), 0, neighbour_idxs.reshape((-1)))
        outputs = outputs.reshape((-1, self._ncodebooks, self._subvect_len))
        outputs_final = outputs
        with torch.no_grad():
            outputs_final -= inputs
        outputs_final += inputs

        if is_training:
            alpha = 1.0
            beta = self._beta
            gamma = 0.0

            reg = alpha * torch.mean((outputs - inputs.requires_grad_(False))**2)
            reg += beta * torch.mean((outputs.requires_grad_(False) - inputs)**2)
            reg += gamma * torch.mean(torch.mean(-response, [0]))

        return codes, outputs_final
