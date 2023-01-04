'''
@file dpq_nn.py
@author Sciroccogti (scirocco_gti@yeah.net)
@brief 
@date 2022-12-31 12:41:49
@modified: 2023-01-04 16:27:39
'''

import numpy as np
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"


class DPQNetwork(torch.nn.Module):
    def __init__(self, ncentroids: int, ncodebooks: int, subvect_len: int,
                 centroids: np.ndarray,
                 tie_in_n_out: bool = True, query_metric: str = "dot", shared_centroids: bool = False,
                 beta: float = 0.0, tau: float = 1.0, softmax_BN: bool = True) -> None:
        '''
        :param ncentroids: K
        :param ncodebooks: C in mithral, or D in kpq
        :param subvect_len: D/C in mithral, or d_in, d_out in kpq
        :param centroids: (ncodebooks, ncentroids, subvect_len)
        :param query_metric: "euclidean" or "dot"
        '''
        if shared_centroids or beta != 0.0:
            raise NotImplementedError(
                "shared_centroids, beta haven't been implemented yet")
        assert np.shape(centroids) == (ncodebooks, ncentroids, subvect_len)
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
        self._tau = Parameter(torch.tensor(
            [1.0], device=device), requires_grad=False)  # constant
        self._softmax_BN = softmax_BN
        self._centroids_k = Parameter(torch.from_numpy(centroids).float().to(device))
        if self._tie_in_n_out:
            self._centroids_v = self._centroids_k
        else:
            self._centroids_v = Parameter(torch.zeros_like(self._centroids_k))
        if self._softmax_BN:
            self.bn = torch.nn.BatchNorm1d(self._ncodebooks, device=device)

    def forward(self, inputs: torch.Tensor, is_training: bool = True
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        :param inputs: (batch_size, ncodebooks, subvect_len)
        '''
        if type(inputs) != torch.Tensor:
            inputs = torch.from_numpy(inputs).to(device).requires_grad_()
        assert inputs.size(1) == self._ncodebooks and inputs.size(2) == self._subvect_len,\
            f"inputs must be (batch_size, ncodebooks, subvect_len), current is {inputs.size()}"
        if inputs.dtype != torch.float:
            inputs = inputs.float()
        # centroids_k = self._centroids_k  # (C, K, subvect_len)
        # centroids_v = self._centroids_v

        # TODO: other metrics
        if self._query_metric == "euclidean":
            # should be negative in order to select the closest via max
            response = -torch.cdist(inputs.transpose(0, 1), self._centroids_k, 2).transpose(0, 1)
        elif self._query_metric == "dot":
            # (C, bs, subvect_len) * (C, subvect_len, K) = (C, bs, K)
            response = torch.matmul(inputs.transpose(0, 1), self._centroids_k.transpose(1, 2))
            # (C, bs, K) -> (bs, C, K)
            response = response.transpose(0, 1)
        else:
            raise NotImplementedError
        assert(response.size(1) == self._ncodebooks and response.size(2) == self._ncentroids)

        # TODO: softmax_BN
        if self._softmax_BN:
            response = self.bn(response)
        response_prob = torch.softmax(response / self._tau, dim=-1)

        mse, codes = torch.max(response, -1)  # (batch_size, C)

        # neighbour_idxs = codes

        if self._tie_in_n_out:
            if not self._shared_centroids:
                D_base = torch.from_numpy(
                    np.array([self._ncentroids*d for d in range(self._ncodebooks)])
                ).to(device)
                neighbour_idxs = D_base + codes
            else:
                neighbour_idxs = codes
            # centroids_v = centroids_v.reshape((-1, self._subvect_len))# (C, K, subvect_len)
            outputs = torch.index_select(
                self._centroids_v.reshape((-1, self._subvect_len)), 0, neighbour_idxs.reshape((-1)))
            outputs = outputs.reshape((-1, self._ncodebooks, self._subvect_len))
            inputs_nograd = inputs.clone().detach()
            outputs_nograd = outputs.clone().detach()
            outputs_final = outputs_nograd - inputs_nograd + inputs
            if is_training:
                alpha = 1.0
                beta = self._beta
                gamma = 0.0

                reg = alpha * torch.mean((outputs - inputs_nograd)**2)
                reg += beta * torch.mean((outputs_nograd - inputs)**2)
                reg += gamma * torch.mean(torch.mean(-response, [0]))

        else:
            neighbour_idxs = codes
            nb_idxs_onehot = torch.tensor(
                F.one_hot(neighbour_idxs, self._ncentroids), device=device, requires_grad=True)
            nb_idxs_onehot = response_prob - (response_prob - nb_idxs_onehot).requires_grad_(False)
            outputs = torch.matmul(nb_idxs_onehot.transpose(0, 1), self._centroids_v)
            outputs_final = outputs.transpose(0, 1)

            if is_training:
                beta = self._beta
                reg = - beta * torch.mean(
                    torch.mean(nb_idxs_onehot * torch.log(response_prob + 1e-10), dim=2))

        return codes, mse, self._centroids_v
