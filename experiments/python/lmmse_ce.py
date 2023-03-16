'''
@file lmmse_ce.py
@author Sciroccogti (scirocco_gti@yeah.net)
@brief 
@date 2023-03-15 12:26:08
@modified: 2023-03-16 13:08:35
'''

import numpy as np
from dft_main import Transceiver, cal_NMSE, convert_complexToReal_Y
from tqdm import tqdm
import csv


class LMMSE(Transceiver):
    def __init__(self, params):
        super().__init__(params)

    def LMMSEChannelEst(self, Ypilot, beta, snr):
        H_LS = Ypilot / self.Xpilot

        # Calculate weighting matrix based on transmitted pilots and noise variance
        W_lmmse = np.zeros((self.Ncarrier, self.Ncarrier), dtype=complex)
        Rp = np.cov(H_LS, rowvar=False)
        W_lmmse = Rp @ np.linalg.inv(Rp + (beta / snr) * np.eye(self.Ncarrier))
        H_est = W_lmmse @ H_LS
        return H_est

    def FER(self, outputPath: str):
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
                InfoStream = self.Bit_create(int(Bitlen * self.ldpc_rate))
                BitStream = InfoStream
                X = np.zeros((1, self.Ncarrier), dtype=complex)
                for nf in range(self.Ncarrier):
                    X[0, nf] = self.Modulation(BitStream[0, 2 * nf:2 * nf + 2])
                # 生成信道矩阵，DFT信道估计
                H = self.Channel_create(0)
                noise = np.random.randn(
                    self.Ncarrier, 1)+1j * np.random.randn(self.Ncarrier, 1)
                Ypilot = np.dot(H, self.Xpilot) + np.sqrt(sigma_2/2) * noise
                # Hest_DFT = H # 测试
                Hest_DFT, nmse_dft, nmse_idft, h_nmse = self.Channel_est(
                    Ypilot, dft_est=dft_est, idft_est=idft_est)

                rawh_nmse = cal_NMSE(convert_complexToReal_Y(
                    H), convert_complexToReal_Y(Hest_DFT))

                # 更新
                NMSE_dft[0][i] += nmse_dft
                NMSE_idft[0][i] += nmse_idft
                H_NMSE[0][i] += h_nmse
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
