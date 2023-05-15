'''
@file lmmse_ce.py
@author Sciroccogti (scirocco_gti@yeah.net)
@brief 
@date 2023-03-15 12:26:08
@modified: 2023-05-06 20:51:57
'''

import csv
import os
import time

import numpy as np
from amm_methods import *
from dft_main import (IEEE802_11_model, Transceiver, cal_NMSE,
                      convert_complexToReal_Y)
from tqdm import tqdm
from scipy import interpolate
from matmul import _estimator_for_method_id

def nmse(X: np.ndarray, X_hat: np.ndarray):
    diff = X - X_hat
    if len(diff.shape) < 2:
        diff = np.expand_dims(diff, axis=1)
    ret = np.matmul(diff.conj().T, diff)
    return np.real(np.squeeze(ret))


class LMMSE(Transceiver):
    def __init__(self, params):
        self.params = params
        self.nCP = params['nCP']
        self.Ncarrier = params['Ncarrier']  # 导频子载波数
        self.qAry = params['qAry']
        self.pilotLoc = params['pilotLoc']
        self.Symbol_num = params['Symbol_num']
        self.matmul_method = params['matmul_method']
        self.ldpc_rate = params['ldpc_rate']
        self.quantize_lut = params['quantize_lut']
        self.bitpilot = np.squeeze(self.Bit_create(
            self.qAry * self.Ncarrier * self.Symbol_num))  # 列向量
        self.Xpilot = np.zeros((len(self.pilotLoc)), dtype=complex)  # 调制后的导频
        for nf in range(len(self.pilotLoc)):
            self.Xpilot[nf] = self.Modulation(self.bitpilot[2 * nf:2 * nf + 2])

    def LMMSEChannelEst(self, Rhh: np.ndarray, H_LS: np.ndarray, snr: float) -> np.ndarray:
        """
        :param Rhh: 信道协方差矩阵，为对角阵
        """
        # beta = E(x^2) * E(1 / x^2)，x 为星座点
        if self.qAry in [1, 2]:
            beta = 1
        elif self.qAry == 4:
            beta = 17 / 9
        elif self.qAry == 6:
            beta = 2.6854170765732626
        else:
            assert False, "qAry must be 1, 2, 4, 6"

        # Calculate weighting matrix based on transmitted pilots and noise variance
        # W_lmmse = Rhh @ np.linalg.inv(Rhh + (beta / snr) * np.eye(len(Rhh)))

        RDS = getRDS(H_LS)
        W = self.getLFC(RDS, snr)

        H_est = W @ H_LS
        return H_est

    def LMMSErf2(self, H: np.ndarray) -> np.ndarray:
        tau_rms = getRDS(H)
        df = 1 / self.Ncarrier
        K3 = np.repeat(np.array([[i for i in range(len(H))]]).T, len(H), axis=1)
        K4 = np.repeat(np.array([[i for i in range(len(H))]]), len(H), axis=0)

        rf2 = 1 / (1 + 2j * np.pi * tau_rms * df * (self.Ncarrier / len(H)) * (K3 - K4))
        return rf2

    def getLFC(self, RDS: float, snr: float) -> np.ndarray:
        df = 1 / self.Ncarrier
        Npilot = len(self.pilotLoc)

        K3 = np.repeat(np.array([[i for i in range(Npilot)]]).T, Npilot, axis=1)
        K4 = np.repeat(np.array([[i for i in range(Npilot)]]), Npilot, axis=0)
        rf2 = 1 / (1 + 2j * np.pi * RDS * df * (self.Ncarrier / Npilot) * (K3 - K4))

        W = rf2 @ np.linalg.inv(rf2 + (1 / snr) * np.eye(Npilot))
        return W

    def LSChannelEst(self, Ypilot):
        H_est = np.squeeze(Ypilot / self.Xpilot)
        return np.diag(H_est)
        # cs = interpolate.interp1d(self.pilotLoc, H_est, kind="nearest-up", fill_value="extrapolate")
        # H_LS = cs(np.arange(0, self.Ncarrier))
        # # H_LS = np.interp(np.arange(0, self.Ncarrier), self.pilotLoc, H_est)
        # return np.diag(np.squeeze(H_LS))

    def interp(self, H_pilot):
        cs = interpolate.interp1d(self.pilotLoc, np.diag(H_pilot),
                                  kind="nearest-up", fill_value="extrapolate")
        H_intp = cs(np.arange(0, self.Ncarrier))
        return np.diag(H_intp)

    def estAutoCor(self, H_LS):
        """
        R(dk) = E_k{H_LS(k) * H_LS(k + dk)}
        :param H_LS: LS估计的信道
        :return: 信道自相关估计
        """
        # R0 = H_LS @ H_LS.conj().T  # 对角线即为 H_LS[k] * H_LS[k]，归一化后应当为 R0 = beta

        rf2 = self.LMMSErf2(H_LS)
        R = rf2
        return R

    def train(self, sample: int = 1000):
        SNRs = 10 ** (np.array(self.params["learnSNRs"]) / 10)
        RDSs = np.empty(shape=(sample, 1))
        for i in range(sample):
            H = self.genChannel()
            RDSs[i][0] = getRDS(H)

        hparams_dict = {
            'ncodebooks': self.params["ncodebooks"],
            'ncentroids': self.params["ncentroids"],
            'quantize_lut': self.params["quantize_lut"],
            'nbits': self.params["nbits"],
            'upcast_every': -1,
            'SNRs': SNRs,
            # 'elemwise_dist_func': dists_elemwise_sq,
        }
        est = _estimator_for_method_id(self.params["matmul_method"], **hparams_dict)
        est.fit(RDSs)
        return est

    def sim(self, outputPath: str):
        est = None
        if self.params["matmul_method"] not in ["LS", "LMMSE"]:
            est = self.train(sample=51200)
        SNRs = self.params['SNR']
        BER = np.zeros((len(SNRs)))
        FER = np.zeros((len(SNRs)))
        NMSE_dft = np.zeros((len(SNRs)))
        NMSE_idft = np.zeros((len(SNRs)))
        H_NMSE = np.zeros((len(SNRs)))
        rawH_NMSE = np.zeros((len(SNRs)))
        ErrorFrame = self.params['ErrorFrame']
        TestFrame = self.params['TestFrame']
        Bitlen = self.qAry * self.Ncarrier * self.Symbol_num
        for i, SNRdb in enumerate(SNRs):
            sigma_2 = np.power(10, (-SNRdb/10))
            ns = 0
            print("SNR: ", SNRdb)
            bar = tqdm(range(TestFrame), ncols=100)
            for ns in bar:
                bar.set_description_str("%.2fdB" % SNRdb)
                bar.set_postfix_str("FER: %.2e" % (FER[i] / ns))
                # 生成信息比特、调制
                InfoStream = np.squeeze(self.Bit_create(int(Bitlen * self.ldpc_rate)))
                BitStream = InfoStream
                X = np.zeros((self.Ncarrier), dtype=complex)
                pPilotLoc = 0
                for nf in range(self.Ncarrier):
                    if pPilotLoc < len(self.pilotLoc) and nf == self.pilotLoc[pPilotLoc]:
                        # 插入导频
                        X[nf] = self.Xpilot[pPilotLoc]
                        pPilotLoc += 1
                    else:
                        X[nf] = self.Modulation(BitStream[2 * nf:2 * nf + 2])

                # 生成信道矩阵
                # H = self.Channel_create(0)
                H = self.genChannel()
                noise = np.random.randn(
                    self.Ncarrier) + 1j * np.random.randn(self.Ncarrier)
                Y = np.dot(H, X) + np.sqrt(sigma_2/2) * noise
                Ypilot = Y[self.pilotLoc]
                Hest_LS = self.LSChannelEst(Ypilot)

                if params["matmul_method"] == "LS":
                    Hest_DFT = self.interp(Hest_LS)
                elif params["matmul_method"] == "LMMSE":
                    # R = self.estAutoCor(np.diag(np.diag(H)[self.pilotLoc]))
                    # R = self.estAutoCor(Hest_LS)
                    # Hest_LMMSE = self.LMMSEChannelEst(R, Hest_LS, 2/sigma_2)

                    RDS = getRDS(Hest_LS)
                    snr = 10 ** (SNRdb / 10)
                    W = self.getLFC(RDS, 3*snr)
                    Hest_LMMSE = W @ Hest_LS
                    Hest_DFT = self.interp(Hest_LMMSE)
                elif params["matmul_method"] == "LMMSE_PQ":
                    RDS = getRDS(Hest_LS)
                    W = est.predict(np.array([[RDS]]), SNRdb)[0]
                    Hest_LMMSEPQ = W @ Hest_LS
                    Hest_DFT = self.interp(Hest_LMMSEPQ)
                else:
                    raise NotImplementedError
                # rawh_nmse = nmse(convert_complexToReal_Y(
                #     # H), convert_complexToReal_Y(Hest_DFT))
                #     np.diag(H)[self.pilotLoc]), convert_complexToReal_Y(np.diag(Hest_DFT)[self.pilotLoc]))  # 只算插值前的误差

                rawh_nmse = nmse(np.diag(H)[self.pilotLoc], np.diag(Hest_DFT)[self.pilotLoc])

                # 更新
                # NMSE_dft[i] += nmse_dft
                # NMSE_idft[i] += nmse_idft
                # H_NMSE[i] += h_nmse
                rawH_NMSE[i] += rawh_nmse / len(self.pilotLoc)

                # 均衡、解调
                G = np.dot(np.conj(Hest_DFT.T), np.linalg.inv(
                    Hest_DFT*np.conj(Hest_DFT.T)+sigma_2*np.eye(self.Ncarrier)))
                Xest = np.dot(G, Y)
                Xest = np.transpose(Xest)
                rho = np.diag(np.dot(G, Hest_DFT))
                LLR = np.zeros((BitStream.size))
                for nf in range(self.Ncarrier):
                    miu_k = rho[nf]
                    epsilon_2 = miu_k - miu_k**2
                    LLR[2*nf:2*nf + 2] = self.QPSK_LLR(Xest[nf], miu_k, epsilon_2)
                LLR = np.array([1 if x >= 0 else 0 for x in LLR])
                count_error = 0
                for j in range(InfoStream.size):
                    if j in self.pilotLoc:
                        continue
                    if InfoStream[j] != LLR[j]:
                        count_error += 1
                BER[i] += count_error
                if count_error != 0:
                    FER[i] += 1
                if FER[i] >= ErrorFrame:
                    break
            BER[i] /= (ns + 1) * self.Ncarrier * self.qAry * self.ldpc_rate
            FER[i] /= (ns + 1)
            NMSE_dft[i] /= (ns + 1)
            NMSE_idft[i] /= (ns + 1)
            H_NMSE[i] /= (ns + 1)
            rawH_NMSE[i] /= (ns + 1)

            with open(outputPath, "a+") as fout:
                writer = csv.writer(fout)
                writer.writerow([SNRdb, BER[i], FER[i], NMSE_dft[i],
                                 NMSE_idft[i], H_NMSE[i], rawH_NMSE[i]])

        return BER, FER, NMSE_dft, NMSE_idft, H_NMSE, rawH_NMSE

    def genChannel(self):
        n_paths = self.params["L"]  # 信道路径数
        path_gains_linear = self.params["PathGain"]
        path_gains_matrix = np.diag(path_gains_linear)  # 路径增益矩阵
        Rayleigh = (np.random.randn(n_paths,) + 1j * np.random.randn(n_paths,)) / np.sqrt(2)
        ht = np.dot(path_gains_matrix, Rayleigh)  # 信道系数
        H = np.fft.fft(ht, self.Ncarrier)  # 信道矩阵 (1, Ncarrier)
        H = np.diag(np.squeeze(H))
        return H


def getRDS(H: np.ndarray) -> float:
    H = np.diag(H)
    k = np.array([i for i in range(len(H))])
    HH = H @ H.conj().T
    tmp = H * H.conj() * k
    r = np.sum(tmp) / HH
    r2 = tmp @ k.T / HH
    RDS = np.sqrt(r2 - r**2)
    return float(RDS.real)


def PDP_Pedestrian_B():
    # sample every 100ns
    pdp = [0.0 for _ in range(38)]
    pdp[0] = 10**(0/10)
    pdp[2] = 10**(-0.9/10)
    pdp[8] = 10**(-4.9/10)
    pdp[12] = 10**(-8.0/10)
    pdp[23] = 10**(-7.8/10)
    pdp[37] = 10**(-23.9/10)
    # pdp /= np.sum(pdp)
    return pdp


# def Channel_Pedestrian_A():
#     Ncarrier = 512

#     tau = [0, 200e-9, 800e-9, 1200e-9, 2300e-9, 3700e-9]
#     powerdB = [0, -0.9, -4.9, -8.0, -7.8, -23.9]

#     fs = 5e6
#     freq = np.arange(-fs/2, fs/2, Ncarrier)
#     pdp /= np.sum(pdp)
#     return pdp

_rms = 25e-9

params = {
    'nCP': 64,
    'Ncarrier': 512,
    'qAry': 2,
    'pilotLoc': [i for i in range(1, 512-64, 8)],
    'Symbol_num': 1,
    'ldpc_rate': 1,
    'L': 38,
    # 'PathGain': [0.7432358676242078,0.9453056768750847,0.03936564739705284,0.04485075815177875,0.7474396724970536,0.24430572962343622,0.8110458033559482,0.8293422474904226,0.39356716943821934,0.8027501479321497,0.27315030042606303,0.18789834683016238,0.3941687035467426,0.6888936766683286,0.2435882240357481,0.0008258433002652499],
    # 'PathGain': np.linspace(1, 0.1, 16).tolist(),
    # 'PathGain': np.power(10, [i/10 for i in range(0, -16, -1)]),
    # 'PathGain': IEEE802_11_model(_rms, 50e-9, 6),
    'PathGain': PDP_Pedestrian_B(),
    'SNR': np.linspace(20, -15, 15).tolist(),
    'ErrorFrame': 200,
    'TestFrame': 20000,
    'LDPC_iter': 20,
    'ncodebooks': 1,
    'ncentroids': 3,
    'quantize_lut': False,
    'nbits': 32,
    'learnSNRs': [2.5, 10, 17.5],
    'matmul_method': "LMMSE"
}

if __name__ == "__main__":
    _dir = os.path.dirname(os.path.abspath(__file__))
    starttime = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    foutName = _dir + "/results/lmmse/" + starttime + ".csv"

    doPathDetect = False  # 是否是检测径
    doTrain = False  # 是否是生成训练集

    if doPathDetect:
        results_ = ["h_ests", "FER", "NMSE_dft"]
    else:
        results_ = ["BER", "FER", "NMSE_dft",
                    "NMSE_idft", "H_NMSE", "rawH_NMSE"]

    with open(foutName, "x", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(["start_time", starttime])
        for key, value in params.items():
            if type(value) == list or type(value) == np.ndarray:
                writer.writerow([key] + list(value))
            else:
                writer.writerow([key, value])
        writer.writerow(["SNR"] + results_)

    lmmse = LMMSE(params)
    BER, FER, NMSE_dft, NMSE_idft, H_NMSE, rawH_NMSE = lmmse.sim(foutName)
    print("BER", BER)
    print("FER", FER)
    print("NMSE_dft", NMSE_dft)
    print("NMSE_idft", NMSE_idft)
    print("H_NMSE", H_NMSE)
    print("rawH_NMSE", rawH_NMSE)

    stoptime = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    with open(foutName, "a+") as fout:
        writer = csv.writer(fout)
        writer.writerow(["stop_time", stoptime])
