'''
@file dpq_encoder.py
@author Sciroccogti (scirocco_gti@yeah.net)
@brief 
@date 2022-12-29 13:27:52
@modified: 2023-01-15 22:25:32
'''

from collections.abc import Callable

import numpy as np
import torch
import tqdm
import vquantizers as vq
from dpq.dpq_nn import DPQNetwork
from torch.utils.tensorboard.writer import SummaryWriter

device = "cuda" if torch.cuda.is_available() else "cpu"  # TODO: set in params

_sliceData_lastpos = 0


def sliceData(sample: int, snr: float, X: np.ndarray | None
              ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    global _sliceData_lastpos
    assert X is not None
    _sliceData_lastpos += sample
    return X[_sliceData_lastpos-sample:_sliceData_lastpos], None, None


class DPQEncoder(vq.MultiCodebookEncoder):
    def __init__(
        self, ncodebooks, ncentroids: int = 16,
        quantize_lut=True, nbits=8, upcast_every=-1, accumulate_how='sum',
        genDataFunc: Callable[[int, float, np.ndarray | None],
                              tuple[np.ndarray, np.ndarray | None, np.ndarray | None]] = sliceData,
    ):
        super().__init__(ncodebooks=ncodebooks, ncentroids=ncentroids, quantize_lut=quantize_lut,
                         nbits=nbits, upcast_every=upcast_every, accumulate_how=accumulate_how)
        self.genDataFunc = genDataFunc

    def name(self):
        return "{}_{}".format('DPQ', super().name())

    def params(self):
        return {'ncodebooks': self.ncodebooks,
                'ncentroids': self.ncentroids,
                'quantize_lut': self.quantize_lut,
                'nbits': self.nbits,
                'upcast_every': self.upcast_every,
                }

    def fit(self, X: np.ndarray, Q) -> None:
        '''
        use DPQ to learn centroids

        :param X: left matrix, just used to train
        :param Q: right matrix, not used, should be the same as the online Q
        '''
        # TODO: query_metric, shared_centroids, tau

        # Mithral style
        # self.splits_lists, self.centroids = learn_mithral(
        #     X, self.ncodebooks, ncentroids=self.ncentroids, lut_work_const=self.lut_work_const,
        #     nonzeros_heuristic=self.nonzeros_heuristic)
        writer = SummaryWriter()
        loss_type = "mse"

        # PQ style
        len_PQ = 51200  # TODO
        X_PQ = X[:len_PQ, :]
        self.subvect_len = int(np.ceil(X_PQ.shape[1] / self.ncodebooks))
        # self.centroids = np.random.random(
        #     (self.ncentroids, self.ncodebooks, self.subvect_len)) * 10 - 5
        # return
        X_PQ = vq.ensure_num_cols_multiple_of(X_PQ, self.ncodebooks)
        # encode_algo: None
        # centroids: (K, C, subvect_len)
        self.centroids, tot_sse_using_mean = vq._learn_centroids(
            X_PQ, self.ncentroids, self.ncodebooks, self.subvect_len, return_tot_mse=True)
        # encode_algo: multisplit
        # self.encode_algo = "multisplits"
        # self.splits_lists, self.centroids = clusterize.learn_splits_in_subspaces(
        #     X, subvect_len=self.subvect_len, nsplits_per_subs=self.code_bits,
        #     algo=self.encode_algo)

        print(f"Using {device} device")
        model = DPQNetwork(
            ncentroids=self.ncentroids,
            ncodebooks=self.ncodebooks,
            subvect_len=self.subvect_len,
            centroids=self.centroids.transpose((1, 0, 2)),
            # centroids=np.random.random(
            #     (self.ncodebooks, self.ncentroids, self.subvect_len)) * 10 - 5,
            # tie_in_n_out=False,
            query_metric="euclidean",
            use_EMA=True,
        ).to(device)

        batch_size = 2**12  # TODO
        epoch = 100000
        is_end2end = True
        genEvery = 50

        optimizer = torch.optim.SGD(model.parameters(), lr=0.9)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", factor=0.5, patience=50, verbose=True)
        mse_per_batch = torch.tensor(tot_sse_using_mean / len_PQ * batch_size, device=device)
        print("mse_per_batch ", mse_per_batch)
        print("Press Ctrl+C to stop training and continue")

        bar = tqdm.tqdm(range(0, epoch))
        X_multi, Y_multi, W = self.genDataFunc(batch_size * genEvery, 0.0, X)
        kpq_centroids = None
        for i in bar:
            try:
                if i % genEvery == 0 and i > 0:
                    X_multi, Y_multi, W = self.genDataFunc(batch_size * genEvery, 0.0, X)
                X_single = X_multi[(i % genEvery) * batch_size:(1 + i % genEvery) * batch_size]
                if isinstance(X_single, np.ndarray):
                    inputs = torch.from_numpy(
                        X_single.reshape((-1, self.ncodebooks, self.subvect_len))
                    ).to(device).requires_grad_()
                else:
                    assert isinstance(X_single, torch.Tensor)
                    inputs = X_single.reshape(
                        (-1, self.ncodebooks, self.subvect_len)).requires_grad_()

                optimizer.zero_grad()
                outputs, mse, kpq_centroids = model.forward(inputs, is_training=True)
                if is_end2end:
                    assert Y_multi is not None and W is not None
                    self.centroids = kpq_centroids.transpose(0, 1).cpu().detach().numpy()
                    Y_single = Y_multi[(i % genEvery) * batch_size:(1 + i % genEvery) * batch_size]
                    X_enc = self.encode_X(X_single)
                    luts = self.encode_Q(W.T)
                    Y_est = self.dists_enc(X_enc, luts, self.quantize_lut)
                    loss = np.mean((Y_single - Y_est)**2) / np.var(Y_single)
                else:
                    if loss_type == "ce":
                        lossfn = torch.nn.CrossEntropyLoss()
                        loss = lossfn(inputs, outputs)
                    else:
                        assert loss_type == "mse"
                        tot_mse = torch.sum(mse)
                        loss = tot_mse / mse_per_batch
                writer.add_scalar("Loss/train", loss, i)
                # loss.backward()
                bar.set_description_str("loss={:.3g}".format(loss))
                optimizer.step()
                scheduler.step(loss)
            except KeyboardInterrupt:
                print("Training stopped manually at epoch %d/%d" % (i, epoch))
                break
        if kpq_centroids != None:
            self.centroids = kpq_centroids.transpose(0, 1).cpu().detach().numpy()

    def encode_Q(self, Q: np.ndarray) -> np.ndarray:
        '''
        generate luts using centroids

        :param Q:
        :return luts
        '''
        # PQ style
        Q = np.atleast_2d(Q)
        Q = vq.ensure_num_cols_multiple_of(Q, self.ncodebooks)

        luts = np.zeros((Q.shape[0], self.ncodebooks, self.ncentroids))
        # print("Q shape: ", Q.shape)
        for i, q in enumerate(Q):
            lut = vq._fit_pq_lut(q, centroids=self.centroids,
                                 elemwise_dist_func=vq.dists_elemwise_dot)
            luts[i] = lut.T
        return luts

    def encode_X(self, X: np.ndarray) -> np.ndarray:
        '''
        encode left matrix online, currently all copied from PQEncoder

        :param X: online left matrix
        '''
        # PQ style
        X = vq.ensure_num_cols_multiple_of(X, self.ncodebooks)
        # encode_algo: None
        idxs = vq.pq._encode_X_pq(X, codebooks=self.centroids)
        # encode_algo: multisplit
        # idxs = clusterize.encode_using_splits(X, self.subvect_len, self.splits_lists, "multi")

        # self.offsets is set in MultiCodebookEncoder.__init__
        return idxs + self.offsets
