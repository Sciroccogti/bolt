'''
@file lmmse_ce.py
@author Sciroccogti (scirocco_gti@yeah.net)
@brief 
@date 2023-03-15 12:26:08
@modified: 2023-04-06 22:27:55
'''

import csv
import os
import time

import numpy as np
from amm_methods import *
from dft_main import (IEEE802_11_model, Transceiver, cal_NMSE,
                      convert_complexToReal_Y)
from tqdm import tqdm


class LMMSE(Transceiver):
    def __init__(self, params):
        super().__init__(params)

    def LMMSEChannelEst(self, Rhh, Ypilot, snr):
        # beta = E(x^2) * E(1 / x^2)，x 为星座点
        if self.qAry in [1, 2]:
            beta = 1
        elif self.qAry == 4:
            beta = 17 / 9
        elif self.qAry == 6:
            beta = 2.6854170765732626
        else:
            assert False, "qAry must be 1, 2, 4, 6"
        H_LS = Ypilot / self.Xpilot

        # Calculate weighting matrix based on transmitted pilots and noise variance
        # W_lmmse = np.zeros((self.Ncarrier, self.Ncarrier), dtype=complex)
        W_lmmse = Rhh @ np.linalg.inv(Rhh + (beta / snr) * np.eye(self.Ncarrier))
        H_est = W_lmmse @ H_LS
        return np.diag(np.squeeze(H_est))

    def LSChannelEst(self, Ypilot):
        H_est = Ypilot / self.Xpilot
        return np.diag(np.squeeze(H_est))

    def sim(self, outputPath: str):
        SNRs = self.params['SNR']
        BER = np.zeros((1, len(SNRs)))
        FER = np.zeros((1, len(SNRs)))
        NMSE_dft = np.zeros((1, len(SNRs)))
        NMSE_idft = np.zeros((1, len(SNRs)))
        H_NMSE = np.zeros((1, len(SNRs)))
        rawH_NMSE = np.zeros((1, len(SNRs)))
        ErrorFrame = self.params['ErrorFrame']
        TestFrame = self.params['TestFrame']
        Bitlen = self.qAry * self.Ncarrier * self.Symbol_num
        for i, SNR in enumerate(SNRs):
            sigma_2 = np.power(10, (-SNR/10))
            ns = 0
            print("SNR: ", SNR)
            bar = tqdm(range(TestFrame), ncols=100)
            for ns in bar:
                bar.set_description_str("%.2fdB" % SNR)
                bar.set_postfix_str("FER: %.2e" % (FER[0][i] / ns))
                # 生成信息比特、调制
                InfoStream = self.genBit(int(Bitlen * self.ldpc_rate))
                BitStream = InfoStream
                X = np.zeros((1, self.Ncarrier), dtype=complex)
                for nf in range(self.Ncarrier):
                    X[0, nf] = self.Modulation(BitStream[0, 2 * nf:2 * nf + 2])
                # 生成信道矩阵，DFT信道估计
                H = self.Channel_create(0)
                noise = np.random.randn(
                    self.Ncarrier, 1)+1j * np.random.randn(self.Ncarrier, 1)
                Ypilot = np.dot(H, self.Xpilot) + np.sqrt(sigma_2/2) * noise
                Rhh = np.dot(H, H.conj().T)
                Hest_LMMSE = self.LMMSEChannelEst(Rhh, Ypilot, 1/sigma_2)
                Hest_LS = self.LSChannelEst(Ypilot)

                if params["matmul_method"] == "LS":
                    Hest_DFT = Hest_LS
                else:
                    Hest_DFT = Hest_LMMSE

                rawh_nmse = cal_NMSE(convert_complexToReal_Y(
                    H), convert_complexToReal_Y(Hest_DFT))
                lsh_nmse = cal_NMSE(convert_complexToReal_Y(
                    H), convert_complexToReal_Y(Hest_LS))

                # 更新
                # NMSE_dft[0][i] += nmse_dft
                # NMSE_idft[0][i] += nmse_idft
                # H_NMSE[0][i] += h_nmse
                rawH_NMSE[0][i] += rawh_nmse

                noise = np.random.randn(
                    self.Ncarrier, 1) + 1j * np.random.randn(self.Ncarrier, 1)
                # 均衡、解调
                Y = np.dot(H, np.transpose(X)) + np.sqrt(sigma_2/2) * noise
                G = np.dot(np.conj(Hest_DFT.T), np.linalg.inv(
                    Hest_DFT*np.conj(Hest_DFT.T)+sigma_2*np.eye(self.Ncarrier)))
                Xest = np.dot(G, Y)
                Xest = np.transpose(Xest)
                rho = np.diag(np.dot(G, Hest_DFT))
                LLR = np.zeros((1, BitStream.size))
                for nf in range(self.Ncarrier):
                    miu_k = rho[nf]
                    epsilon_2 = miu_k - miu_k**2
                    LLR[0][2*nf:2*nf +
                           2] = self.QPSK_LLR(Xest[0][nf], miu_k, epsilon_2)
                LLR = np.array([[1 if x >= 0 else 0 for x in LLR[0]]])
                count_error = 0
                for j in range(InfoStream.size):
                    if InfoStream[0][j] != LLR[0][j]:
                        count_error += 1
                BER[0][i] += count_error
                if count_error != 0:
                    FER[0][i] += 1
                if FER[0][i] >= ErrorFrame:
                    break
            BER[0][i] /= (ns + 1) * self.Ncarrier * self.qAry * self.ldpc_rate
            FER[0][i] /= (ns + 1)
            NMSE_dft[0][i] /= (ns + 1)
            NMSE_idft[0][i] /= (ns + 1)
            H_NMSE[0][i] /= (ns + 1)
            rawH_NMSE[0][i] /= (ns + 1)

            with open(outputPath, "a+") as fout:
                writer = csv.writer(fout)
                writer.writerow([SNR, BER[0][i], FER[0][i], NMSE_dft[0][i],
                                 NMSE_idft[0][i], H_NMSE[0][i], rawH_NMSE[0][i]])

        return BER, FER, NMSE_dft, NMSE_idft, H_NMSE, rawH_NMSE


def PDP_Pedestrian_B():
    pdp = [0.0 for _ in range(38)]
    pdp[0] = 10**(0/10)
    pdp[2] = 10**(-0.9/10)
    pdp[8] = 10**(-4.9/10)
    pdp[12] = 10**(-8.0/10)
    pdp[23] = 10**(-7.8/10)
    pdp[37] = 10**(-23.9/10)
    return pdp


_rms = 25e-9

params = {
    'Nifft': 128,
    'Ncarrier': 128,
    'qAry': 2,
    'Symbol_len': 128,
    'Symbol_num': 1,
    'ldpc_rate': 1,
    'L': 38,
    # 'PathGain': [0.7432358676242078,0.9453056768750847,0.03936564739705284,0.04485075815177875,0.7474396724970536,0.24430572962343622,0.8110458033559482,0.8293422474904226,0.39356716943821934,0.8027501479321497,0.27315030042606303,0.18789834683016238,0.3941687035467426,0.6888936766683286,0.2435882240357481,0.0008258433002652499],
    # 'PathGain': np.linspace(1, 0.1, 16).tolist(),
    # 'PathGain': np.power(10, [i/10 for i in range(0, -16, -1)]),
    # 'PathGain': IEEE802_11_model(_rms, 50e-9, 6),
    'PathGain': PDP_Pedestrian_B(),
    'SNR': np.linspace(-20, 15, 15).tolist(),
    'ErrorFrame': 200,
    'TestFrame': 20000,
    'LDPC_iter': 20,
    'ncodebooks': 128,
    'ncentroids': 128,
    'quantize_lut': True,
    'nbits': 32,
    'rms': _rms,
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
