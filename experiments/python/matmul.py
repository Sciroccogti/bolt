#!/bin/env/python

import functools
import numpy as np
import pprint
import scipy
import time

import amm
import matmul_datasets as md
import pyience as pyn
import compress
from amm_methods import *

import sys
from Logger import *
# std_out重定向
# sys.stdout = Logger("./log/caltech/Figure7_log.txt", sys.stdout)
# sys.stderr = Logger("./log/caltech/Figure7_debug.txt", sys.stderr)


import amm_methods as methods

from joblib import Memory
_memory = Memory('.', verbose=0)


# NUM_TRIALS = 1
NUM_TRIALS = 10


# @_memory.cache
def _estimator_for_method_id(method_id, **method_hparams):
    return methods.METHOD_TO_ESTIMATOR[method_id](**method_hparams)


def _ntrials_for_method(method_id, ntasks):
    # return 1 # TODO rm
    if ntasks > 1:  # no need to avg over trials if avging over multiple tasks
        return 1
    # return NUM_TRIALS if method_id in methods.NONDETERMINISTIC_METHODS else 1
    return NUM_TRIALS if method_id in methods.RANDOM_SKETCHING_METHODS else 1


# ================================================================ metrics

def _compute_compression_metrics(ar):
    # if quantize_to_type is not None:
    #     ar = ar.astype(quantize_to_type)
    # ar -= np.min(ar)
    # ar /= (np.max(ar) / 65535)  # 16 bits
    # ar -= 32768  # center at 0
    # ar = ar.astype(np.int16)

    # elem_sz = ar.dtype.itemsize
    # return {'nbytes_raw': ar.nbytes,
    #         'nbytes_blosc_noshuf': len(_blosc_compress(
    #             ar, elem_sz=elem_sz, shuffle=blosc.NOSHUFFLE)),
    #         'nbytes_blosc_byteshuf': len(_blosc_compress(
    #             ar, elem_sz=elem_sz, shuffle=blosc.SHUFFLE)),
    #         'nbytes_blosc_bitshuf': len(_blosc_compress(
    #             ar, elem_sz=elem_sz, shuffle=blosc.BITSHUFFLE)),
    #         'nbytes_zstd': len(_zstd_compress(ar)),
    #         'nbits_cost': nbits_cost(ar).sum() // 8,
    #         'nbits_cost_zigzag':
    #             nbits_cost(zigzag_encode(ar), signed=False).sum() // 8,
    #         'nbytes_sprintz': compress.sprintz_packed_size(ar)
    #         }

    return {'nbytes_raw': ar.nbytes,
            'nbytes_sprintz': compress.sprintz_packed_size(ar)}


def _cossim(Y, Y_hat):
    ynorm = np.linalg.norm(Y) + 1e-20
    yhat_norm = np.linalg.norm(Y_hat) + 1e-20
    return ((Y / ynorm) * (Y_hat / yhat_norm)).sum()


def _compute_metrics(task, Y_hat, compression_metrics=True, **sink):
    Y = task.Y_test
    diffs = Y - Y_hat
    raw_mse = np.mean(diffs * diffs)
    normalized_mse = raw_mse / np.var(Y)
    # Y_meannorm = Y - Y.mean()
    # Y_hat_meannorm = Y_hat - Y_hat.mean()
    # ynorm = np.linalg.norm(Y_meannorm) + 1e-20
    # yhat_norm = np.linalg.norm(Y_hat_meannorm) + 1e-20
    # r = ((Y_meannorm / ynorm) * (Y_hat_meannorm / yhat_norm)).sum()
    metrics = {'raw_mse': raw_mse, 'normalized_mse': normalized_mse,
               'corr': _cossim(Y - Y.mean(), Y_hat - Y_hat.mean()),
               'cossim': _cossim(Y, Y_hat),  # 'bias': diffs.mean(),
               'y_mean': Y.mean(), 'y_std': Y.std(),
               'yhat_std': Y_hat.std(), 'yhat_mean': Y_hat.mean()}
    if compression_metrics:

        # Y_q = compress.quantize(Y, nbits=8)
        # Y_hat_q = compress.quantize(Y_hat, nbits=8)
        # diffs_q = Y_q - Y_hat_q
        # # diffs_q = compress.zigzag_encode(diffs_q).astype(np.uint8)
        # assert Y_q.dtype == np.int8
        # assert diffs_q.dtype == np.int8

        Y_q = compress.quantize(Y, nbits=12)
        Y_hat_q = compress.quantize(Y_hat, nbits=12)
        diffs_q = Y_q - Y_hat_q
        assert Y_q.dtype == np.int16
        assert diffs_q.dtype == np.int16

        # Y_q = quantize_i16(Y)

        # # quantize to 16 bits
        # Y = Y - np.min(Y)
        # Y /= (np.max(Y) / 65535)  # 16 bits
        # Y -= 32768  # center at 0
        # Y = Y.astype(np.int16)
        # diffs =

        metrics_raw = _compute_compression_metrics(Y_q)
        metrics.update({k + '_orig': v for k, v in metrics_raw.items()})
        metrics_raw = _compute_compression_metrics(diffs_q)
        metrics.update({k + '_diffs': v for k, v in metrics_raw.items()})

    if task.info:
        problem = task.info['problem']
        metrics['problem'] = problem
        if problem == 'softmax':
            lbls = task.info['lbls_test'].astype(np.int32)
            b = task.info['biases']
            logits_amm = Y_hat + b
            logits_orig = Y + b
            lbls_amm = np.argmax(logits_amm, axis=1).astype(np.int32)
            lbls_orig = np.argmax(logits_orig, axis=1).astype(np.int32)
            # print("Y_hat shape : ", Y_hat.shape)
            # print("lbls hat shape: ", lbls_amm.shape)
            # print("lbls amm : ", lbls_amm[:20])
            metrics['acc_amm'] = np.mean(lbls_amm == lbls)
            metrics['acc_orig'] = np.mean(lbls_orig == lbls)

        elif problem in ('1nn', 'rbf'):
            lbls = task.info['lbls_test'].astype(np.int32)
            lbls_centroids = task.info['lbls_centroids']
            lbls_hat_1nn = []
            rbf_lbls_hat = []
            W = task.W_test
            centroid_norms_sq = (W * W).sum(axis=0)
            sample_norms_sq = (task.X_test * task.X_test).sum(
                axis=1, keepdims=True)

            k = W.shape[1]
            nclasses = np.max(lbls_centroids) + 1
            affinities = np.zeros((k, nclasses), dtype=np.float32)
            for kk in range(k):
                affinities[kk, lbls_centroids[kk]] = 1

            for prods in [Y_hat, Y]:
                dists_sq_hat = (-2 * prods) + \
                    centroid_norms_sq + sample_norms_sq
                # 1nn classification
                centroid_idx = np.argmin(dists_sq_hat, axis=1)
                lbls_hat_1nn.append(lbls_centroids[centroid_idx])
                # rbf kernel classification (bandwidth=1)
                # gamma = 1. / np.sqrt(W.shape[0])
                # gamma = 1. / W.shape[0]
                gamma = 1
                similarities = scipy.special.softmax(
                    -dists_sq_hat * gamma, axis=1)
                class_probs = similarities @ affinities
                rbf_lbls_hat.append(np.argmax(class_probs, axis=1))

            lbls_amm_1nn, lbls_orig_1nn = lbls_hat_1nn
            rbf_lbls_amm, rbf_lbls_orig = rbf_lbls_hat
            metrics['acc_amm_1nn'] = np.mean(lbls_amm_1nn == lbls)
            metrics['acc_orig_1nn'] = np.mean(lbls_orig_1nn == lbls)
            metrics['acc_amm_rbf'] = np.mean(rbf_lbls_amm == lbls)
            metrics['acc_orig_rbf'] = np.mean(rbf_lbls_orig == lbls)

            if problem == '1nn':
                lbls_amm, lbls_orig = rbf_lbls_amm, rbf_lbls_orig
            elif problem == 'rbf':
                lbls_amm, lbls_orig = rbf_lbls_amm, rbf_lbls_orig

            orig_acc_key = 'acc-1nn-raw'
            if orig_acc_key in task.info:
                metrics[orig_acc_key] = task.info[orig_acc_key]

            metrics['acc_amm'] = np.mean(lbls_amm == lbls)
            metrics['acc_orig'] = np.mean(lbls_orig == lbls)
        elif problem == 'sobel':
            assert Y.shape[1] == 2
            grad_mags_true = np.sqrt((Y * Y).sum(axis=1))
            grad_mags_hat = np.sqrt((Y_hat * Y_hat).sum(axis=1))
            diffs = grad_mags_true - grad_mags_hat
            metrics['grad_mags_nmse'] = (
                (diffs * diffs).mean() / grad_mags_true.var())
        elif problem.lower().startswith('dog'):
            # difference of gaussians
            assert Y.shape[1] == 2
            Z = Y[:, 0] - Y[:, 1]
            Z_hat = Y_hat[:, 0] - Y_hat[:, 1]
            diffs = Z - Z_hat
            metrics['dog_nmse'] = (diffs * diffs).mean() / Z.var()

    return metrics


# ================================================================ driver funcs

def _eval_amm(task, est, fixedB=True, **metrics_kwargs):
    est.reset_for_new_task()
    if fixedB:
        est.set_B(task.W_test)

    # print("eval_amm validating task: ", task.name)
    # task.validate(train=False, test=True)
    # print(f"task {task.name} matrix hashes:")
    # pprint.pprint(task._hashes())

    # print("task: ", task.name)
    # print("X_test shape: ", task.X_test.shape)
    # print("W_test shape: ", task.W_test.shape)
    t = time.perf_counter()
    # Y_hat = est.predict(task.X_test.copy(), task.W_test.copy())
    Y_hat = est.predict(task.X_test, task.W_test)
    # Y_hat = task.X_test @ task.W_test  # yep, zero error
    duration_secs = time.perf_counter() - t

    metrics = _compute_metrics(task, Y_hat, **metrics_kwargs)
    metrics['secs'] = duration_secs
    # metrics['nmultiplies'] = est.get_nmuls(task.X_test, task.W_test)
    metrics.update(est.get_speed_metrics(
        task.X_test, task.W_test, fixedB=fixedB))

    # print("eval_amm re-validating task: ", task.name)
    # task.validate(train=False, test=True)
    # print(f"task {task.name} matrix hashes:")
    # pprint.pprint(task.hashes())

    return metrics


# @functools.lru_cache(maxsize=None)
# @_memory.cache
def _fitted_est_for_hparams(method_id, hparams_dict, X_train, W_train,
                            Y_train, **kwargs):
    est = _estimator_for_method_id(method_id, **hparams_dict)
    est.fit(X_train, W_train, Y=Y_train, **kwargs)
    return est


def estFactory(methods=['Mithral'], ntasks=1, ncodebooks=32, ncentroids=256,
               verbose=1, limit_ntasks=-1, tasks_all_same_shape=False,
               X_path="", W_path="", Y_path="", dir=""):
    methods = methods.DEFAULT_METHODS if methods is None else methods
    if isinstance(methods, str):
        methods = [methods]
    if limit_ntasks is None or limit_ntasks < 1:
        limit_ntasks = np.inf
    # method_id
    method_id = methods[0]
    # hparams_dict
    # nc: 1 2 4 8 16 32
    # lut: 2 4 -1
    if (METHOD_SCALAR_QUANTIZE in methods):
        hparams_dict = {}
    elif (METHOD_HASHJL in methods) or (METHOD_SVD in methods) or (METHOD_FD_AMM in methods):
        hparams_dict = {'d': 2}
    elif (METHOD_SPARSE_PCA in methods):
        dvals = [1, 2, 4, 8, 16, 32, 64]
        alpha_vals = (1. / 16384, .03125, .0625, .125, .25, .5, 1, 2, 4, 8)
        hparams_dict = [{'d': d, 'alpha': alpha}
                        for d in dvals for alpha in alpha_vals][0]
    elif (METHOD_MITHRAL in methods):
        hparams_dict = {'ncodebooks': ncodebooks, 'lut_work_const': -1} # Mithral 的 ncentroids 自动生成
    else:
        hparams_dict = {'ncodebooks': ncodebooks, 'ncentroids': ncentroids}

    est = None

    if verbose > 0:
        print("==============================")
        print("running method: ", method_id)

    # for hparams_dict in _hparams_for_method(method_id):
    if verbose > 2:
        print("got hparams: ")
        pprint.pprint(hparams_dict)

    try:
        prev_X_shape, prev_Y_shape = None, None
        prev_X_std, prev_Y_std = None, None
        # est = None
        for i, task in enumerate(md.load_dft_train(X_path, W_path, Y_path, dir)):
            if i + 1 > limit_ntasks:
                raise StopIteration()
            if verbose > 3:
                print("-------- running task: {} ({}/{})".format(
                    task.name, i + 1, ntasks))
                task.validate_shapes()  # fail fast if task is ill-formed

            can_reuse_est = (
                (i != 0) and (est is not None)
                and (prev_X_shape is not None)
                and (prev_Y_shape is not None)
                and (prev_X_std is not None)
                and (prev_Y_std is not None)
                and (task.X_train.shape == prev_X_shape)
                and (task.Y_train.shape == prev_Y_shape)
                and (task.X_train.std() == prev_X_std)
                and (task.Y_train.std() == prev_Y_std))

            # can_reuse_est = False # TODO
            if not can_reuse_est:
                try:
                    task.W_train = np.atleast_2d(task.W_train)
                    est = _fitted_est_for_hparams(
                        method_id, hparams_dict,
                        task.X_train, task.W_train, task.Y_train)
                except amm.InvalidParametersException as e:
                    # hparams don't make sense for task (eg, D < d)
                    print(f"hparams apparently invalid: {e}")
                    if verbose > 2:
                        print(f"hparams apparently invalid: {e}")
                    est = None
                    if tasks_all_same_shape:
                        raise StopIteration()
                    else:
                        continue

                prev_X_shape = task.X_train.shape
                prev_Y_shape = task.Y_train.shape
                prev_X_std = task.X_train.std()
                prev_Y_std = task.Y_train.std()

    except StopIteration:  # no more tasks for these hparams
        pass

    return est


def eval_matmul(est, X_test=None, W_test=None):
    # task = list(enumerate(md.load_dft_test()))[0][1]
    # task = list(enumerate(md.construct_dft_test(X_test, W_test))[0][1])
    # tast = md.load_dft_tasks()

    est.reset_for_new_task()
    est.set_B(W_test)
    Y_hat = est.predict(X_test, W_test)
    return Y_hat


if __name__ == '__main__':
    np.set_printoptions(formatter={'float': lambda f: "{:.2f}".format(f)},
                        linewidth=100)
    est = estFactory()
    Y_hat = eval_matmul(est)
    print(Y_hat)
