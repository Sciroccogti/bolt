import csv
import math
import os
import time

import matmul as mm
import numpy as np
import tensorflow as tf
import torch
from amm_methods import *
from sionna.fec.ldpc.decoding import LDPC5GDecoder, LDPC5GEncoder
from tqdm import tqdm
from scipy.linalg import toeplitz

# from tensorflow.python.ops.numpy_ops import np_config


# np_config.enable_numpy_behavior() # enable tensor.size() for tensorflow


class Transceiver:
    def __init__(self, params):
        self.params = params
        self.Nifft = params['Nifft']
        self.Ncarrier = params['Ncarrier']  # 导频子载波数
        self.qAry = params['qAry']
        self.Symbol_len = params['Symbol_len']
        self.Symbol_num = params['Symbol_num']
        self.matmul_method = params['matmul_method']
        self.ldpc_rate = params['ldpc_rate']
        self.quantize_lut = params['quantize_lut']

        self.bitpilot = self.Bit_create(
            self.qAry * self.Ncarrier * self.Symbol_num)  # 列向量
        self.Xpilot = np.zeros((1, self.Ncarrier), dtype=complex)  # 调制后的导频
        for nf in range(self.Ncarrier):
            self.Xpilot[0, nf] = self.Modulation(
                self.bitpilot[0, 2 * nf:2 * nf + 2])
        self.Xpilot = np.transpose(self.Xpilot)  # (Ncarries, 1)
        self.Create_DFTmatrix()

    def Create_DFTmatrix(self):
        '''Init DFTm'''
        n = np.arange(self.Nifft).reshape(1, self.Nifft)
        k = np.arange(self.Nifft).reshape(1, self.Nifft)
        Wn = np.exp(-1j * 2 * np.pi / self.Nifft)
        nk = np.dot(n.T, k)
        self.DFTm = np.zeros(nk.shape, dtype=complex)
        for i in range(self.DFTm.shape[0]):
            for j in range(self.DFTm.shape[1]):
                self.DFTm[i][j] = np.power(Wn, nk[i][j])
        self.IDFTm = 1/self.Nifft * np.conj(self.DFTm)

    def Bit_create(self, length: int):
        '''生成一帧信息比特/导频'''
        bitstream = np.random.randint(0, 2, (1, length))
        return bitstream

    def Encoder(self):
        # 编码，暂无
        return

    def Modulation(self, input_frame):
        # 暂时只能判断一个星座点，未改好
        # input_frame : input bit stream (0,1)
        # qAry: 1--bpsk ; 2--qpsk ; 4--16qam ; 6--64qam
        if self.qAry == 1:
            BPSK_I = [-1, 1]
            QAM_input_I = BPSK_I[input_frame]
            output_modu = QAM_input_I
        elif self.qAry == 2:
            QPSK_IQ = [-1, 1]
            QAM_input_I = QPSK_IQ[input_frame[0]]
            QAM_input_Q = QPSK_IQ[input_frame[1]]
            output_modu = (QAM_input_I + 1j * QAM_input_Q) / np.sqrt(2)
        elif self.qAry == 4:
            QAM_16_IQ = [-3, -1, 3, 1]
            QAM_input_I = QAM_16_IQ[input_frame[0]*2 + input_frame[1]]
            QAM_input_Q = QAM_16_IQ[input_frame[2]*2 + input_frame[3]]
            output_modu = (QAM_input_I + 1j * QAM_input_Q) / np.sqrt(10)
        elif self.qAry == 6:
            QAM_64_IQ = [-7, -5, -1, -3, 7, 5, 1, 3]
            QAM_input_I = QAM_64_IQ[input_frame[0] *
                                    4 + input_frame[1]*2 + input_frame[2]]
            QAM_input_Q = QAM_64_IQ[input_frame[3] *
                                    4 + input_frame[4]*2 + input_frame[5]]
            output_modu = (QAM_input_I + 1j * QAM_input_Q) / np.sqrt(42)
        return output_modu

    def Channel_create(self, corr: float) -> np.ndarray:
        # Correlation-based stochastic model
        # 定义信道参数
        n_tx = 4  # 发射天线数
        n_rx = 4  # 接收天线数
        n_paths = self.params["L"]  # 信道路径数
        corr_tx = corr  # 发射端相关系数
        corr_rx = corr  # 接收端相关系数

        # # 路径增益（dB）
        # path_gains = np.array([3, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15])
        path_gains_linear = self.params["PathGain"]
        path_gains_matrix = np.diag(path_gains_linear)  # 路径增益矩阵
        if corr == 0.0:
            noise = (np.random.randn(n_paths,) + 1j * np.random.randn(n_paths,))
            ht = np.dot(np.sqrt(path_gains_matrix/2), noise)
            H = np.fft.fft(ht, self.Nifft)  # (1, Nifft)
            H = np.diag(np.squeeze(H))  # (Nifft, Nifft)
        else:
            # 生成相关矩阵
            corr_matrix_tx = exp_corr_mat(corr_tx, n_tx)  # 发射端相关矩阵
            corr_matrix_rx = exp_corr_mat(corr_rx, n_rx)  # 接收端相关矩阵
            corr_matrix = np.kron(corr_matrix_tx, corr_matrix_rx)  # 信道相关矩阵 (MIMO 的 H)
            # 生成相关信道路径增益
            n_samples = 1  # 采样数
            # path_gains_linear = 10**(path_gains/10)  # 路径增益（线性）
            # channel_path_gains = np.zeros((n_tx*n_rx, n_samples))  # 信道路径增益矩阵
            # for i in range(n_samples):
            #     noise = (np.random.randn(n_paths,) + 1j * np.random.randn(n_paths,))  # 生成高斯噪声
            #     # 计算信道路径增益
            #     channel_path_gains[:, i] = np.dot(corr_matrix, np.dot(path_gains_matrix, noise))

            noise = (np.random.randn(n_paths,) + 1j * np.random.randn(n_paths,))
            ht = np.dot(np.sqrt(path_gains_matrix/2), noise)
            hh = corr_matrix * ht
            H = np.fft.fft(hh, n=self.Nifft, axis=1)
            H = np.diag(np.squeeze(H))  # (Nifft, Nifft)
        return H

    def Channel_est(self, Ypilot, dft_est, idft_est):
        Hest = Ypilot/self.Xpilot
        # h_est = np.fft.ifft(np.transpose(Hest),self.Nifft)
        # 变换到时域后，只保留前 L 个，以消除其它时延上的噪声
        h_est, NMSE_idft = self.IDFT(
            np.transpose(Hest), self.Nifft, est=idft_est)
        h_DFTfilter = h_est
        # h_DFTfilter = h_est[0][0:20] # 在self.IDFT中截取
        # Hest_DFTfilter = np.fft.fft(h_DFTfilter,self.Nifft)
        Hest_DFTfilter, NMSE_dft = self.DFT(
            h_DFTfilter, self.Nifft, est=dft_est)
        Hest_DFTfilter = np.diag(np.squeeze(Hest_DFTfilter))
        ################################################################
        H_NMSE = 0
        if self.matmul_method != METHOD_EXACT:
            h_est_p = self.IDFT_i(np.transpose(Hest), self.Nifft, est=idft_est)
            h_DFTfilter_p = h_est_p
            Hest_DFTfilter_p = self.DFT_i(
                h_DFTfilter_p, self.Nifft, est=dft_est)
            Hest_DFTfilter_p = np.diag(np.squeeze(Hest_DFTfilter_p))

            H_NMSE = cal_NMSE(convert_complexToReal_Y(
                Hest_DFTfilter_p), convert_complexToReal_Y(Hest_DFTfilter))

        return Hest_DFTfilter, NMSE_dft, NMSE_idft, H_NMSE

    def Channel_Splitest(self, Ypilot, dft_est, idft_ests_, slice: int = 4):
        Hest = Ypilot/self.Xpilot
        # h_est = np.fft.ifft(np.transpose(Hest),self.Nifft)
        h_est, NMSE_idft = self.IDFTSplit(
            np.transpose(Hest), self.Nifft, ests_=idft_ests_, slice=slice)
        h_DFTfilter = h_est
        # h_DFTfilter = h_est[0][0:20] # 在self.IDFT中截取
        # Hest_DFTfilter = np.fft.fft(h_DFTfilter,self.Nifft)
        Hest_DFTfilter, NMSE_dft = self.DFT(
            h_DFTfilter, self.Nifft, est=dft_est)
        Hest_DFTfilter = np.diag(np.squeeze(Hest_DFTfilter))
        ################################################################
        H_NMSE = 0
        if self.matmul_method != METHOD_EXACT:
            h_est_p = self.IDFT_i(np.transpose(Hest), self.Nifft)
            h_DFTfilter_p = h_est_p
            Hest_DFTfilter_p = self.DFT_i(
                h_DFTfilter_p, self.Nifft, est=dft_est)
            Hest_DFTfilter_p = np.diag(np.squeeze(Hest_DFTfilter_p))

            H_NMSE = cal_NMSE(convert_complexToReal_Y(
                Hest_DFTfilter_p), convert_complexToReal_Y(Hest_DFTfilter))

        return Hest_DFTfilter, NMSE_dft, NMSE_idft, H_NMSE

    def DFT(self, xn, N, est=None):
        # 代替fft
        W = self.DFTm[0:xn.size]
        NMSE_dft = 0
        xp = np.dot(xn, W)
        if self.matmul_method != METHOD_EXACT:
            xn = convert_complexToReal_X(xn)
            W = convert_complexToReal_W(W)
            Xk_tmp = mm.eval_matmul(est, xn, W)
            Xk = covert_realToComplex_Y(Xk_tmp)
            NMSE_dft = cal_NMSE(Xk_tmp, convert_complexToReal_Y(xp))
        else:
            # Exact
            assert self.matmul_method == METHOD_EXACT, "Other methods not supported!"
            Xk = xp  # TODO
        return Xk, NMSE_dft

    def IDFT(self, Xk, N, est=None):
        # 代替ifft
        W = self.IDFTm[:, 0:16]  # 此处已经截取
        NMSE_idft = 0
        xp = np.dot(Xk, W)
        if self.matmul_method != METHOD_EXACT:
            Xk = convert_complexToReal_X(Xk)
            W = convert_complexToReal_W(W)
            xn_tmp = mm.eval_matmul(est, Xk, W)[:, 0:16]
            xn = covert_realToComplex_Y(xn_tmp)
            NMSE_idft = cal_NMSE(xn_tmp, convert_complexToReal_Y(xp))
        else:
            # Exact
            assert self.matmul_method == METHOD_EXACT, "Other methods not supported!"
            xn = xp  # TODO
        return xn, NMSE_idft

    def IDFTSplit(self, Xk, N, ests_: list, slice: int = 4):
        """
        代替ifft
        [A1 A2 A3 A4] * [B1; B2; B3; B4] = [A1B1 + A2B2 + A3B3 + A4B4]
        """
        W = self.IDFTm[:, 0:20]  # 此处已经截取
        NMSE_idft = 0
        xp = np.dot(Xk, W)

        sliceLen = Xk.shape[1] // slice
        assert(Xk.shape[1] == 128)

        if self.matmul_method != METHOD_EXACT:
            xn_tmp = np.zeros((Xk.shape[0] * 2, W.shape[1]))
            for i in range(slice):
                XkSplit = convert_complexToReal_X(
                    Xk[:, i * sliceLen: (i+1) * sliceLen])
                WSplit = convert_complexToReal_W(
                    W[i * sliceLen: (i+1) * sliceLen])
                xn_tmp += mm.eval_matmul(ests_[i], XkSplit, WSplit)
            xn = covert_realToComplex_Y(xn_tmp)
            NMSE_idft = cal_NMSE(xn_tmp, convert_complexToReal_Y(xp))
        else:
            # Exact
            assert self.matmul_method == METHOD_EXACT, "Other methods not supported!"
            xn = xp  # TODO
        return xn, NMSE_idft

    def DFT_i(self, xn, N, est=None):
        # 代替fft
        W = self.DFTm[0:xn.size]
        xp = np.dot(xn, W)
        # Exact
        Xk = xp  # TODO

        return Xk

    def IDFT_i(self, Xk, N, est=None):
        # 代替ifft
        W = self.IDFTm[:, 0:20]  # 此处已经截取
        xp = np.dot(Xk, W)
        # Exact
        xn = xp  # TODO

        return xn

    def QPSK_LLR(self, est_x, miu, epsilon_2):
        bit = [0, 0, 0, 1, 1, 0, 1, 1]
        signal = np.zeros((1, 4), dtype=complex)
        LLR = [0, 0]
        term = [0, 0, 0, 0]
        for i in range(4):
            signal[0, i] = self.Modulation(bit[2 * i:2 * i + 2])
        epsilon_2 = max(0.001, epsilon_2)
        for i in range(4):
            term[i] = np.exp(-np.abs(est_x - miu*signal[0][i])**2/(epsilon_2))
        group = np.reshape(bit, (2, 4), order='F')
        P0_1, P0_2, P1_1, P1_2 = 0, 0, 0, 0
        for i in range(4):
            if group[0][i] == 0:
                P0_1 += term[i]
            else:
                P1_1 += term[i]
            if group[1][i] == 0:
                P0_2 += term[i]
            else:
                P1_2 += term[i]
        if P0_1 == 0:
            LLR[0] = float('inf')
        elif P1_1 == 0:
            LLR[0] = float('-inf')
        else:
            LLR[0] = math.log10(P1_1 / P0_1)
        if P0_2 == 0:
            LLR[1] = float('inf')
        elif P1_2 == 0:
            LLR[1] = float('-inf')
        else:
            LLR[1] = math.log10(P1_2 / P0_2)
        return LLR

    def decoder(self):
        #  解码，暂无
        return

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
        if self.ldpc_rate < 1:
            encoder = LDPC5GEncoder(Bitlen * self.ldpc_rate, Bitlen, dtype=tf.int64)
            decoder = LDPC5GDecoder(
                encoder=encoder, num_iter=self.params['LDPC_iter'], hard_out=True)

        dft_est = None
        idft_est = None
        if self.matmul_method != METHOD_EXACT:
            dft_est = mm.estFactory(methods=[METHOD_EXACT], verbose=3,  # TODO: change to matmul_method
                                    ncodebooks=self.params["ncodebooks"],
                                    ncentroids=self.params["ncentroids"],
                                    X_path="DFT_X.npy", W_path="DFT_W.npy", Y_path="DFT_Y.npy",
                                    dir="dft", quantize_lut=self.quantize_lut)
            idft_est = mm.estFactory(methods=[self.matmul_method],
                                     ncodebooks=self.params["ncodebooks"],
                                     ncentroids=self.params["ncentroids"],
                                     X_path="IDFT_X.npy", W_path="IDFT_W.npy", Y_path="IDFT_Y.npy",
                                     dir="dft", nbits=self.params["nbits"],
                                     quantize_lut=self.quantize_lut,
                                     lut_work_const=-1,
                                     genDataFunc=self.gen_IDFTTrain,
                                     )
        else:
            assert self.matmul_method == METHOD_EXACT, "Other methods not supported!"
        for i, SNR in enumerate(SNRs):
            sigma_2 = np.power(10, (-SNR/10))
            # sigma_2 = 0 # back-to-back
            ns = 0
            print("SNR: ", SNR)
            bar = tqdm(range(TestFrame), ncols=100)
            for ns in bar:
                bar.set_description_str("%.2fdB" % SNR)
                bar.set_postfix_str("FER: %.2e" % (FER[0][i] / ns))
                # 生成信息比特、调制
                InfoStream = self.Bit_create(int(Bitlen * self.ldpc_rate))
                if self.ldpc_rate < 1:
                    BitStream = encoder(InfoStream).numpy()
                else:
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
                if self.ldpc_rate < 1:
                    LLR = decoder(LLR)
                else:
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

    def SplitFER(self, outputPath: str, slice: int = 4):
        """
        把 IDFT 中的乘法改为分块矩阵乘法，以期提高精确度
        效果与增加码本数完全一致
        """
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

        dft_est = None
        idft_ests_ = []
        if self.matmul_method != METHOD_EXACT:
            dft_est = mm.estFactory(methods=[self.matmul_method], verbose=3,
                                    ncodebooks=self.params["ncodebooks"],
                                    ncentroids=self.params["ncentroids"],
                                    X_path="DFT_X.npy", W_path="DFT_W.npy", Y_path="DFT_Y.npy", dir="dftSplit%d" % slice)
            for i in range(slice):
                idft_ests_.append(mm.estFactory(methods=[self.matmul_method],
                                                ncodebooks=self.params["ncodebooks"],
                                                ncentroids=self.params["ncentroids"],
                                                X_path="IDFT_X%d.npy" % i, W_path="IDFT_W%d.npy" % i, Y_path="IDFT_Y%d.npy" % i, dir="dftSplit%d" % slice))
        else:
            assert self.matmul_method == METHOD_EXACT, "Other methods not supported!"
        for i, SNR in enumerate(SNRs):
            sigma_2 = np.power(10, (-SNR/10))
            # sigma_2 = 0 # back-to-back
            ns = 0
            print("SNR: ", SNR)
            while FER[0][i] < ErrorFrame or ns < TestFrame:
                ns += 1
                # 生成信息比特、调制
                BitStream = self.Bit_create(int(Bitlen * self.ldpc_rate))
                X = np.zeros((1, self.Ncarrier), dtype=complex)
                for nf in range(self.Ncarrier):
                    X[0, nf] = self.Modulation(BitStream[0, 2 * nf:2 * nf + 2])
                # 生成信道矩阵，DFT信道估计
                H = self.Channel_create()
                noise = np.random.randn(
                    self.Ncarrier, 1)+1j * np.random.randn(self.Ncarrier, 1)
                Ypilot = np.dot(H, self.Xpilot) + np.sqrt(sigma_2/2) * noise
                # Hest_DFT = H # 测试
                Hest_DFT, nmse_dft, nmse_idft, h_nmse = self.Channel_Splitest(
                    Ypilot, dft_est=dft_est, idft_ests_=idft_ests_, slice=slice)

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
                LLR = np.array([1 if x >= 0 else 0 for x in LLR[0]])
                count_error = 0
                for j in range(BitStream.size):
                    if BitStream[0][j] != LLR[j]:
                        count_error += 1
                BER[0][i] += count_error
                if count_error != 0:
                    FER[0][i] += 1
            BER[0][i] /= (ns + 1) * self.Ncarrier * self.qAry
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

    def gen_IDFTTrain(self, sample: int, SNR: float, inputs: np.ndarray | None
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        for DPQ to generate training data

        :param inputs: just placeholder, not used
        :return X, Y, W
        """
        IDFT_Xtrain = np.zeros((sample, self.Nifft * 2), dtype=float)
        IDFT_Ytrain = np.zeros((sample, 16), dtype=float)
        IDFT_W = self.IDFTm[:, 0:16]  # 128*20
        sigma_2 = np.power(10, (-SNR / 10))
        for i in range(sample//2):
            H = self.Channel_create()
            noise = np.random.randn(self.Ncarrier, 1) + \
                1j * np.random.randn(self.Ncarrier, 1)
            Ypilot = np.dot(H, self.Xpilot) + np.sqrt(sigma_2 / 2) * noise
            Hest = Ypilot / self.Xpilot
            Xk = np.transpose(Hest)
            IDFT_Xtrain[i*2] = np.concatenate([Xk.real, -Xk.imag], 1)
            IDFT_Xtrain[i*2 + 1] = np.concatenate([Xk.imag, Xk.real], 1)
            xn = np.dot(Xk, IDFT_W)
            IDFT_Ytrain[i*2] = xn.real
            IDFT_Ytrain[i*2 + 1] = xn.imag
        return IDFT_Xtrain, IDFT_Ytrain, np.concatenate([IDFT_W.real, IDFT_W.imag], 0)

    def Channel_createTorch(self, device: str) -> torch.Tensor:
        L = self.params['L']
        try:
            self.PathGainTorchSqrt
        except:
            self.PathGainTorchSqrt = torch.sqrt(
                torch.tensor(self.params["PathGain"], dtype=torch.double, device=device))
        ht = self.PathGainTorchSqrt * (0.5 ** 0.5) *\
            (torch.randn((1, L), device=device) + 1j * torch.randn((1, L), device=device))
        H = torch.fft.fft2(ht, [1, self.Nifft])
        H = torch.diag(H.squeeze())
        return H

    def gen_IDFTTrainTorch(self, sample: int, SNR: float, inputs: np.ndarray | None) -> torch.Tensor:
        """
        for DPQ to generate training data directly in cuda tensor
        actually not faster than CPU version

        :param inputs: just placeholder, not used
        """
        device = "cuda"

        IDFT_Xtrain = torch.zeros((sample, self.Nifft), dtype=torch.complex64, device=device)
        # IDFT_W = torch.tensor(self.IDFTm[:, 0:20], device=device)
        sigma_2 = 10 ** (-SNR / 10)
        s = (sigma_2 / 2) ** 0.5
        try:
            self.XpilotTorch
        except:
            # (1, Nifft)
            self.XpilotTorch = torch.tensor(self.Xpilot, device=device).transpose(0, 1)

        # noise = torch.randn((sample, self.Nifft), device=device) + 1j * \
        #     torch.randn((sample, self.Nifft), device=device)
        # Ypilot = torch.matmul(self.XpilotTorch.expand_as(noise), H) + s * noise  # (sample, Nifft)
        # Hest = Ypilot / self.XpilotTorch.expand_as(noise)
        # IDFT_Xtrain = Hest

        for i in range(sample):
            H = self.Channel_createTorch(device)  # (Nifft, Nifft)
            noise = torch.randn((1, self.Nifft), device=device) + 1j * \
                torch.randn((1, self.Nifft), device=device)
            Ypilot = torch.matmul(self.XpilotTorch, H) + s * noise
            Hest = Ypilot / self.XpilotTorch
            # Xk = torch.transpose(Hest, 0, 1)
            IDFT_Xtrain[i] = Hest

        return IDFT_Xtrain

    def create_Traindata(self, SNR):
        sample = 512000
        DFT_Xtrain = np.zeros((sample, 20), dtype=complex)
        DFT_Ytrain = np.zeros((sample, self.Nifft), dtype=complex)
        DFT_W = self.DFTm[0:20]  # 20*128

        IDFT_Xtrain = np.zeros((sample, self.Nifft), dtype=complex)
        IDFT_Ytrain = np.zeros((sample, 20), dtype=complex)
        IDFT_W = self.IDFTm[:, 0:20]  # 128*20
        sigma_2 = np.power(10, (-SNR / 10))
        for i in range(sample):
            H = self.Channel_create(0)
            noise = np.random.randn(self.Ncarrier, 1) + \
                1j * np.random.randn(self.Ncarrier, 1)
            Ypilot = np.dot(H, self.Xpilot) + np.sqrt(sigma_2 / 2) * noise
            Hest = Ypilot / self.Xpilot
            Xk = np.transpose(Hest)
            IDFT_Xtrain[i] = Xk
            xn = np.dot(Xk, IDFT_W)
            IDFT_Ytrain[i] = xn
            # label = np.fft.ifft(Xk,self.Nifft)

            DFT_Xtrain[i] = xn
            Xk1 = np.dot(xn, DFT_W)
            DFT_Ytrain[i] = Xk1
            # label1 = np.fft.fft(xn,self.Nifft)

        save_mat(convert_complexToReal_X(DFT_Xtrain), "DFT_X.npy")
        save_mat(convert_complexToReal_Y(DFT_Ytrain), "DFT_Y.npy")
        save_mat(convert_complexToReal_W(DFT_W), "DFT_W.npy")

        save_mat(convert_complexToReal_X(IDFT_Xtrain), "IDFT_X.npy")
        save_mat(convert_complexToReal_Y(IDFT_Ytrain), "IDFT_Y.npy")
        save_mat(convert_complexToReal_W(IDFT_W), "IDFT_W.npy")

    def create_SplitIDFTTraindata(self, slice: int = 4):
        """
        split IDFT from 2*128 x 128*20 to 2*32 x 32*20 (slice = 4 = 128/32)
        """
        sample = 25000
        DFT_Xtrain = np.zeros((sample, 20), dtype=complex)
        DFT_Ytrain = np.zeros((sample, 128), dtype=complex)
        DFT_W = self.DFTm[0:20]  # 20*128

        IDFT_Xtrain = np.zeros((sample, 128), dtype=complex)
        IDFT_Ytrain = np.zeros((sample, 20), dtype=complex)
        IDFT_W = self.IDFTm[:, 0:20]  # 128*20
        sigma_2 = np.power(10, (-10 / 10))
        for i in range(sample):
            H = self.Channel_create()
            noise = np.random.randn(self.Ncarrier, 1) + \
                1j * np.random.randn(self.Ncarrier, 1)
            Ypilot = np.dot(H, self.Xpilot) + np.sqrt(sigma_2 / 2) * noise
            Hest = Ypilot / self.Xpilot
            Xk = np.transpose(Hest)
            IDFT_Xtrain[i] = Xk
            xn = np.dot(Xk, IDFT_W)
            IDFT_Ytrain[i] = xn
            # label = np.fft.ifft(Xk,self.Nifft)

            DFT_Xtrain[i] = xn
            Xk1 = np.dot(xn, DFT_W)
            DFT_Ytrain[i] = Xk1
            # label1 = np.fft.fft(xn,self.Nifft)

        save_mat(convert_complexToReal_X(DFT_Xtrain), "DFT_X.npy")
        save_mat(convert_complexToReal_Y(DFT_Ytrain), "DFT_Y.npy")
        save_mat(convert_complexToReal_W(DFT_W), "DFT_W.npy")

        sliceLen = 128 // slice
        assert(slice * sliceLen == 128)
        for i in range(slice):
            save_mat(convert_complexToReal_X(
                IDFT_Xtrain[:, i * sliceLen: (i+1) * sliceLen]), "IDFT_X%d.npy" % i)
            save_mat(convert_complexToReal_Y(
                IDFT_Ytrain[i * sliceLen: (i+1) * sliceLen]), "IDFT_Y%d.npy" % i)
            save_mat(convert_complexToReal_W(
                IDFT_W[i * sliceLen: (i+1) * sliceLen]), "IDFT_W%d.npy" % i)

    def create_SplitBothTraindata(self, slice: int = 4):
        """
        split IDFT from 2*128 x 128*20 to 2*32 x 32*20 (slice = 4 = 128/32)
        split DFT from 2*40 x 40*128 to 2*10 x 
        """
        sample = 25000
        DFT_Xtrain = np.zeros((sample, 20), dtype=complex)
        DFT_Ytrain = np.zeros((sample, 128), dtype=complex)
        DFT_W = self.DFTm[0:20]  # 20*128

        IDFT_Xtrain = np.zeros((sample, 128), dtype=complex)
        IDFT_Ytrain = np.zeros((sample, 20), dtype=complex)
        IDFT_W = self.IDFTm[:, 0:20]  # 128*20
        sigma_2 = np.power(10, (-10 / 10))
        for i in range(sample):
            H = self.Channel_create()
            noise = np.random.randn(self.Ncarrier, 1) + \
                1j * np.random.randn(self.Ncarrier, 1)
            Ypilot = np.dot(H, self.Xpilot) + np.sqrt(sigma_2 / 2) * noise
            Hest = Ypilot / self.Xpilot
            Xk = np.transpose(Hest)
            IDFT_Xtrain[i] = Xk
            xn = np.dot(Xk, IDFT_W)
            IDFT_Ytrain[i] = xn
            # label = np.fft.ifft(Xk,self.Nifft)

            DFT_Xtrain[i] = xn
            Xk1 = np.dot(xn, DFT_W)
            DFT_Ytrain[i] = Xk1
            # label1 = np.fft.fft(xn,self.Nifft)

        save_mat(convert_complexToReal_X(DFT_Xtrain), "DFT_X.npy")
        save_mat(convert_complexToReal_Y(DFT_Ytrain), "DFT_Y.npy")
        save_mat(convert_complexToReal_W(DFT_W), "DFT_W.npy")

        sliceLen = 128 // slice
        assert(slice * sliceLen == 128)
        for i in range(slice):
            save_mat(convert_complexToReal_X(
                IDFT_Xtrain[:, i * sliceLen: (i+1) * sliceLen]), "IDFT_X%d.npy" % i)
            save_mat(convert_complexToReal_Y(
                IDFT_Ytrain[i * sliceLen: (i+1) * sliceLen]), "IDFT_Y%d.npy" % i)
            save_mat(convert_complexToReal_W(
                IDFT_W[i * sliceLen: (i+1) * sliceLen]), "IDFT_W%d.npy" % i)

    def pathDetect(self, outputPath: str):
        SNRs = self.params['SNR']
        BER = np.zeros((1, len(SNRs)))
        FER = np.zeros((1, len(SNRs)))
        NMSE_dft = np.zeros((1, len(SNRs)))
        NMSE_idfts = np.zeros((1, len(SNRs)))
        H_NMSE = np.zeros((1, len(SNRs)))
        rawH_NMSE = np.zeros((1, len(SNRs)))
        h_ests = np.zeros(
            (1, len(SNRs), self.params["L"]), dtype=np.complex128)
        ErrorFrame = self.params['ErrorFrame']
        TestFrame = self.params['TestFrame']
        Bitlen = self.qAry * self.Ncarrier * self.Symbol_num

        dft_est = None
        idft_est = None
        if self.matmul_method != METHOD_EXACT:
            dft_est = mm.estFactory(methods=[self.matmul_method], verbose=3,
                                    ncodebooks=self.params["ncodebooks"],
                                    ncentroids=self.params["ncentroids"],
                                    X_path="DFT_X.npy", W_path="DFT_W.npy", Y_path="DFT_Y.npy", dir="dft")
            idft_est = mm.estFactory(methods=[self.matmul_method],
                                     ncodebooks=self.params["ncodebooks"],
                                     ncentroids=self.params["ncentroids"],
                                     X_path="IDFT_X.npy", W_path="IDFT_W.npy", Y_path="IDFT_Y.npy", dir="dft")
        else:
            assert self.matmul_method == METHOD_EXACT, "Other methods not supported!"
        for i, SNR in enumerate(SNRs):
            sigma_2 = np.power(10, (-SNR/10))
            ns = 0
            print("SNR: ", SNR)
            while FER[0][i] < ErrorFrame or ns < TestFrame:
                ns += 1
                # 生成信息比特、调制
                BitStream = self.Bit_create(int(Bitlen * self.ldpc_rate))
                X = np.zeros((1, self.Ncarrier), dtype=complex)
                for nf in range(self.Ncarrier):
                    X[0, nf] = self.Modulation(BitStream[0, 2 * nf:2 * nf + 2])
                # 生成信道矩阵，DFT信道估计
                H = self.Channel_create()
                noise = np.random.randn(
                    self.Ncarrier, 1)+1j * np.random.randn(self.Ncarrier, 1)
                Ypilot = np.dot(H, self.Xpilot) + np.sqrt(sigma_2/2) * noise

                Hest = Ypilot/self.Xpilot
                # h_est = np.fft.ifft(np.transpose(Hest), self.Nifft)
                h_est, NMSE_idft = self.IDFT(
                    np.transpose(Hest), self.Nifft, est=idft_est)

                h_dists = []
                for j in range(len(self.params["PathGain"])):
                    # 计算 h_est 到各单位向量的欧式距，从而将其分类到对应径
                    pg = np.zeros((self.params["L"]), dtype=np.complex128)
                    pg[j] = 1
                    h_dists.append(np.linalg.norm(np.abs(h_est) - pg))
                pgIdx = np.argmin(h_dists)  # 记录对应径的下标

                # 初始 PathGain 的最大值不在 pgIdx
                if (np.argmax(self.params["PathGain"]) != pgIdx):
                    FER[0][i] += 1

                h_ests[0][i] += h_est[0]
                NMSE_idfts[0][i] += NMSE_idft

            h_ests[0][i] /= (ns + 1)
            FER[0][i] /= (ns + 1)
            NMSE_idfts[0][i] /= (ns + 1)

            with open(outputPath, "a+") as fout:
                writer = csv.writer(fout)
                writer.writerow([SNR, h_ests[0][i], FER[0][i], NMSE_dft[0][i]])

        return FER, NMSE_idfts


def save_mat(mat, fname):
    # fpath = os.path.join(NEW_DIR, fname)
    np.save(fname, mat)


def convert_complexToReal_X(X: np.ndarray):
    """
    |X.real,  -X.imag|

    |X.imag,   X.real|
    """
    X_real = X.real
    X_img = X.imag
    return np.concatenate([
        np.concatenate([X_real, -X_img], 1),
        np.concatenate([X_img, X_real], 1)
    ], 0)


def convert_complexToReal_W(W):
    W_real = W.real
    W_img = W.imag
    return np.concatenate([W_real, W_img], 0)


def convert_complexToReal_Y(Y):
    Y_real = Y.real
    Y_img = Y.imag
    return np.concatenate([Y_real, Y_img], 0)


def covert_realToComplex_Y(Y):
    size = Y.shape[0]
    half_size = (int)(size / 2)
    return Y[: half_size] + 1j * Y[half_size:]

# 仅支持实数的


def cal_NMSE(A, A_hat):
    diffs = A - A_hat
    raw_mse = np.mean(diffs * diffs)
    normalized_mse = raw_mse / np.var(A)
    return normalized_mse


def exp_corr_mat(a: float | complex, n: int):
    assert(np.abs(a) < 1 and a != 0)
    exp = np.arange(n)
    # First column of R
    col = np.power(a, exp)
    # First row of R
    row = np.conj(col)
    r = toeplitz(col, row)
    return r


def IEEE802_11_model(rms: float, Ts: float, L: int) -> np.ndarray:
    '''
    :param rms :Root Mean Square Delay Spread，增大 rms 时延扩展可以降低频域相干性
    :param Ts: sampling time
    :param L: path num
    '''
    assert rms > 0
    # # num of Path
    lmax = max(math.ceil(10 * rms / Ts) + 1, L)
    # power of the first tap
    # sigma0_2 = (1 - math.exp(-Ts / rms)) / (1 - math.exp(-(lmax) * Ts / rms))
    sigma0_2 = 1
    PDP = np.array([sigma0_2 * math.exp(- l * Ts / rms) for l in range(lmax)])
    # cut to L paths
    ret = PDP[:L]  # / sum(PDP[:L])
    return ret


_rms = 500e-9

params = {
    'Nifft': 128,
    'Ncarrier': 128,
    'qAry': 2,
    'Symbol_len': 128,
    'Symbol_num': 1,
    'ldpc_rate': 1,
    'L': 16,
    # 'PathGain': np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    # 'PathGain': [0.7432358676242078,0.9453056768750847,0.03936564739705284,0.04485075815177875,0.7474396724970536,0.24430572962343622,0.8110458033559482,0.8293422474904226,0.39356716943821934,0.8027501479321497,0.27315030042606303,0.18789834683016238,0.3941687035467426,0.6888936766683286,0.2435882240357481,0.0008258433002652499],
    # 'PathGain': np.linspace(1, 0.1, 16).tolist(),
    # 'PathGain': np.power(10, [i/10 for i in range(0, -16, -1)]),
    'PathGain': IEEE802_11_model(_rms, 50e-9, 16),
    'SNR': np.linspace(-20, 15, 15).tolist(),
    'ErrorFrame': 200,
    'TestFrame': 20000,
    'LDPC_iter': 20,
    'ncodebooks': 128,
    'ncentroids': 64,
    'quantize_lut': True,
    'nbits': 16,
    'rms': _rms,
    'matmul_method': METHOD_MITHRAL
}

if __name__ == '__main__':
    _dir = os.path.dirname(os.path.abspath(__file__))
    starttime = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    foutName = _dir + "/results/dft_main_" + starttime + ".csv"

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

    myTransceiver = Transceiver(params)

    if doTrain:
        myTransceiver.create_Traindata(10.0)
    elif not doPathDetect:
        BER, FER, NMSE_dft, NMSE_idft, H_NMSE, rawH_NMSE = myTransceiver.FER(
            foutName)
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
    else:  # pathDetect
        params["PathGain"] = np.array(
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        FER, NMSE_idft = myTransceiver.pathDetect(foutName)
        print("FER", FER)
        print("NMSE_idft", NMSE_idft)

        stoptime = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        with open(foutName, "a+") as fout:
            writer = csv.writer(fout)
            writer.writerow(["stop_time", stoptime])
