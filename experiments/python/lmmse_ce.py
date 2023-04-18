'''
@file lmmse_ce.py
@author Sciroccogti (scirocco_gti@yeah.net)
@brief 
@date 2023-03-15 12:26:08
@modified: 2023-04-18 13:24:05
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

def mse(X:np.ndarray, X_hat:np.ndarray):
    return np.mean(np.abs(X - X_hat)**2)


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
            self.Xpilot[nf] = self.Modulation(
                self.bitpilot[2 * nf:2 * nf + 2])

    def LMMSEChannelEst(self, Rhh, H_LS, snr):
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

        # Rhh = np.diag(Rhh)

        # Calculate weighting matrix based on transmitted pilots and noise variance
        # W_lmmse = np.zeros((self.Ncarrier, self.Ncarrier), dtype=complex)
        W_lmmse = Rhh @ np.linalg.inv(Rhh + (beta / snr) * np.eye(len(Rhh)))

        rf2 = self.LMMSErf2(H_LS)
        W = self.LMMSELFC(rf2, snr)

        H_est = W_lmmse @ H_LS
        return H_est

    def LMMSELFC(self, rf2, snr):
        W = rf2 / (rf2 + 1 / snr)
        return W

    def LMMSErf2(self, H):
        H = np.diag(H)
        k = np.array([i for i in range(len(H))])
        HH = H @ H.conj().T
        tmp = H * H.conj() * k
        r = np.sum(tmp) / HH
        r2 = tmp @ k.T / HH
        tau_rms = np.sqrt(r2 - r**2)
        df = 1 / self.Ncarrier
        K3 = np.repeat(np.array([[i for i in range(len(H))]]).T, len(H), axis=1)
        K4 = np.repeat(np.array([[i for i in range(len(H))]]), len(H), axis=0)

        rf2 = 1 / (1 + 2j * np.pi * tau_rms * df * (self.Ncarrier / len(H)) * (K3 - K4))
        return rf2

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
        R(dk) = E_k{H_LS(k) * H_LS(k + dk}
        :param H_LS: LS估计的信道
        :return: 信道自相关估计
        """
        R0 = H_LS @ H_LS.conj().T # 对角线即为 R0 = H_LS[k] * H_LS[k]
        # R0 = 
        # return Rhh

        # N = len(H_LS)
        # temp = np.correlate(H_LS, H_LS, mode='full')
        # temp = temp[-1::-1] / N
        # Rhh = np.zeros((N, N), dtype=complex)
        # for i in range(N):
        #     Rhh[i, :] = temp[N - 1 - i:N * 2 - 1 - i]
        # return Rhh

        # H_LS = np.diag(H_LS)
        # N = len(H_LS)
        # R = np.zeros((N, N), dtype=complex)
        # for dk in range(N):
        #     for k in range(N):
        #         R[dk, k] = H_LS[k] * np.conj(H_LS)[(k + dk) % N]
        # Rhh = np.mean(R, axis=1)
        # return Rhh

        rf2 = self.LMMSErf2(H_LS)
        R = rf2
        return R


    def sim(self, outputPath: str):
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
        for i, SNR in enumerate(SNRs):
            sigma_2 = np.power(10, (-SNR/10))
            ns = 0
            print("SNR: ", SNR)
            bar = tqdm(range(TestFrame), ncols=100)
            for ns in bar:
                bar.set_description_str("%.2fdB" % SNR)
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
                # noise = np.zeros_like(noise)
                Y = np.dot(H, X) + np.sqrt(sigma_2/2) * noise
                Ypilot = Y[self.pilotLoc]
                # Rhh = np.dot(H, H.conj().T)
                Hest_LS = self.LSChannelEst(Ypilot)

                if params["matmul_method"] == "LS":
                    Hest_DFT = self.interp(Hest_LS)
                else:
                    # R = self.estAutoCor(np.diag(np.diag(H)[self.pilotLoc]))
                    R = self.estAutoCor(Hest_LS)
                    Hest_LMMSE = self.LMMSEChannelEst(R, Hest_LS, 1/sigma_2)
                    Hest_DFT = self.interp(Hest_LMMSE)
                rawh_nmse = mse(convert_complexToReal_Y(
                    # H), convert_complexToReal_Y(Hest_DFT))
                    np.diag(H)[self.pilotLoc]), convert_complexToReal_Y(np.diag(Hest_DFT)[self.pilotLoc]))  # 只算插值前的误差

                # 更新
                # NMSE_dft[i] += nmse_dft
                # NMSE_idft[i] += nmse_idft
                # H_NMSE[i] += h_nmse
                rawH_NMSE[i] += rawh_nmse

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
                writer.writerow([SNR, BER[i], FER[i], NMSE_dft[i],
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
    'SNR': np.linspace(15, -20, 15).tolist(),
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
