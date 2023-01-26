'''
@file dpq_nn.py
@author Sciroccogti (scirocco_gti@yeah.net)
@brief 
@date 2022-12-31 12:41:49
@modified: 2023-01-26 12:52:31
'''

import numpy as np
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"


class DPQNetwork(torch.nn.Module):
    def __init__(self, ncentroids: int, ncodebooks: int, subvect_len: int,
                 W: np.ndarray, centroids: np.ndarray,
                 tie_in_n_out: bool = True, query_metric: str = "dot", shared_centroids: bool = False,
                 beta: float = 0.0, tau: float = 1.0, softmax_BN: bool = False,
                 use_EMA: bool = True,
                 ) -> None:
        '''
        :param ncentroids: K
        :param ncodebooks: C in mithral, or D in kpq
        :param subvect_len: D/C in mithral, or d_in, d_out in kpq
        :param centroids: (ncodebooks, ncentroids, subvect_len)
        :param query_metric: "euclidean" or "dot"

        :param use_EMA: use Exponential Moving Average to update centroids, or use regularization
        '''
        assert np.shape(centroids) == (ncodebooks, ncentroids, subvect_len)
        assert np.shape(W)[0] == ncodebooks*subvect_len and len(np.shape(W)) == 2
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
        self._tau = Parameter(torch.tensor([1.0], device=device), requires_grad=False)
        self._softmax_BN = softmax_BN

        # (C, K, subvect_len)
        self._centroids_k = Parameter(torch.from_numpy(
            centroids).float().to(device), requires_grad=False)
        if self._tie_in_n_out:
            self._centroids_v = self._centroids_k
        else:
            self._centroids_v = Parameter(torch.zeros_like(self._centroids_k), requires_grad=False)
        if self._softmax_BN:
            self.bn = torch.nn.BatchNorm1d(self._ncodebooks, device=device)
        minaxis = [0, 1] if self._shared_centroids else [0]
        self.minaxis = torch.tensor(minaxis, device=device, requires_grad=False)
        self.counts = None
        self.use_EMA = use_EMA

        self.fc = torch.nn.Linear(np.shape(W)[0], np.shape(W)[1], bias=False, device=device)
        self.fc.weight.data = torch.from_numpy(W.T).float().to(device)

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

        if self._softmax_BN:
            response = self.bn(response)
        response_prob = torch.softmax(response / self._tau, dim=-1)

        neg_mse, codes = torch.max(response, -1)  # codes: (batch_size, C)

        # TODO: sampling
        # neighbour_idxs = codes
        if not self._shared_centroids:
            D_base = torch.from_numpy(
                np.array([self._ncentroids*d for d in range(self._ncodebooks)])
            ).to(device)
            neighbour_idxs = D_base + codes
        else:
            neighbour_idxs = codes

        # (bs, C, K)
        nb_idxs_onehot = F.one_hot(codes, num_classes=self._ncentroids).float()
        if self._tie_in_n_out:
            # outputs are nearest centroids of inputs
            outputs = torch.index_select(
                self._centroids_v.reshape((-1, self._subvect_len)), 0, neighbour_idxs.reshape((-1)))
            outputs = outputs.reshape((-1, self._ncodebooks, self._subvect_len))
            inputs_nograd = inputs.clone().detach()
            outputs_nograd = outputs.clone().detach()
            outputs_final = outputs_nograd - inputs_nograd + inputs

            if is_training:
                if not self.use_EMA:
                    alpha = 1.0
                    beta = self._beta
                    gamma = 0.0

                    # TODO: still don't know how to set as a Parameter
                    reg = (alpha * torch.mean((outputs - inputs_nograd) ** 2)
                           + beta * torch.mean((outputs_nograd - inputs)**2)
                           + gamma * torch.mean(torch.min(-response, self.minaxis))).clone().detach().requires_grad_()
                else:
                    # http://arxiv.org/abs/1803.03382 equation9
                    # do a Exponential Moving Average on centroids
                    decay = torch.tensor(0.999, device=device, requires_grad=False)
                    # counts (C, K): the number of inputs that take centroid[k] as its nearest
                    new_counts = torch.sum(nb_idxs_onehot, dim=0).detach()
                    if self.counts == None:
                        self.counts = new_counts
                    else:
                        self.counts = decay * self.counts + (1 - decay) * new_counts
                    # (C, K, subvect_len)
                    inputs_each_cent = torch.sum(
                        torch.matmul(
                            nb_idxs_onehot.reshape(-1, self._ncodebooks, self._ncentroids, 1),
                            inputs.reshape(-1, self._ncodebooks, 1, self._subvect_len)
                        ), dim=0).detach()
                    new_centroids = torch.divide(
                        inputs_each_cent, self.counts.unsqueeze(dim=-1).expand(inputs_each_cent.shape))
                    new_centroids[new_centroids != new_centroids] = 0  # set where 0 / 0 = nan to 0
                    self._centroids_k *= decay
                    self._centroids_k += (1 - decay) * new_centroids

        else:
            nb_idxs_onehot = response_prob - (response_prob - nb_idxs_onehot).detach()
            outputs = torch.matmul(nb_idxs_onehot.transpose(0, 1), self._centroids_v)
            outputs_final = outputs.transpose(0, 1)

            if is_training:
                beta = self._beta
                reg = - beta * torch.mean(
                    torch.mean(nb_idxs_onehot * torch.log(response_prob + 1e-10), dim=2))

        product = self.fc(outputs_final.reshape(-1, self._ncodebooks*self._subvect_len))

        return product, -neg_mse, codes

    def get_data(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._centroids_k, self.fc.weight.data
