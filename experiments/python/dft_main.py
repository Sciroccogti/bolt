import math
import os
import time
from pprint import pprint
from tqdm import trange, tqdm

import numpy as np
import tensorflow as tf
from sionna.fec.ldpc.decoding import LDPC5GDecoder, LDPC5GEncoder
# from tensorflow.python.ops.numpy_ops import np_config

import matmul as mm
from amm_methods import *

# np_config.enable_numpy_behavior() # enable tensor.size() for tensorflow


class Transceiver:
    def __init__(self, params):
        self.params = params
        self.Nifft = params['Nifft']
        self.Ncarrier = params['Ncarrier']
        self.qAry = params['qAry']
        self.Symbol_len = params['Symbol_len']
        self.Symbol_num = params['Symbol_num']
        self.matmul_method = params['matmul_method']
        self.ldpc_rate = params['ldpc_rate']

        self.bitpilot = self.Bit_create(
            self.qAry * self.Ncarrier * self.Symbol_num)  # 列向量
        self.Xpilot = np.zeros((1, self.Ncarrier), dtype=complex)  # 调制后的导频
        for nf in range(self.Ncarrier):
            self.Xpilot[0, nf] = self.Modulation(
                self.bitpilot[0, 2 * nf:2 * nf + 2])
        self.Xpilot = np.transpose(self.Xpilot)
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

    def Channel_create(self):
        L = self.params['L']
        PathGain = self.params['PathGain']
        # PathGain = PathGain/sum(PathGain)
        ht = np.sqrt(PathGain) * (np.sqrt(1 / 2) *
                                  (np.random.randn(1, L) + 1j * np.random.randn(1, L)))
        H = np.fft.fft(ht, self.Nifft)
        H = np.diag(np.squeeze(H))
        return H

    def Channel_est(self, Ypilot, dft_est, idft_est):
        Hest = Ypilot/self.Xpilot
        # h_est = np.fft.ifft(np.transpose(Hest),self.Nifft)
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
            xn_tmp = mm.eval_matmul(est, Xk, W)
            xn = covert_realToComplex_Y(xn_tmp)
            NMSE_idft = cal_NMSE(xn_tmp, convert_complexToReal_Y(xp))
        else:
            # Exact
            assert self.matmul_method == METHOD_EXACT, "Other methods not supported!"
            xn = xp  # TODO
        return xn, NMSE_idft

    def IDFTSplit(self, Xk, N, ests_: list = None, slice: int = 4):
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

    def FER(self):
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
        encoder = LDPC5GEncoder(Bitlen * self.ldpc_rate, Bitlen, dtype=tf.int64)
        decoder = LDPC5GDecoder(encoder=encoder, num_iter=20, hard_out=True)

        dft_est = None
        idft_est = None
        if self.matmul_method != METHOD_EXACT:
            dft_est = mm.estFactory(methods=[METHOD_EXACT], verbose=3, # TODO: change to matmul_method
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
            # sigma_2 = 0 # back-to-back
            ns = 0
            print("SNR: ", SNR)
            bar = tqdm(range(TestFrame))
            for ns in bar:
                bar.set_description_str("%.2fdB" % SNR)
                bar.set_postfix_str("FER: %.2e" % (FER[0][i] / ns))
                # 生成信息比特、调制
                InfoStream = self.Bit_create(int(Bitlen * self.ldpc_rate))
                BitStream = encoder(InfoStream).numpy()
                X = np.zeros((1, self.Ncarrier), dtype=complex)
                for nf in range(self.Ncarrier):
                    X[0, nf] = self.Modulation(BitStream[0, 2 * nf:2 * nf + 2])
                # 生成信道矩阵，DFT信道估计
                H = self.Channel_create()
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
                # LLRhard = np.array([1 if x >= 0 else 0 for x in LLR[0]])
                LLR = decoder(LLR)
                count_error = 0
                for j in range(InfoStream.size):
                    if InfoStream[0][j] != LLR[0][j]:
                        count_error += 1
                BER[0][i] += count_error
                if count_error != 0:
                    FER[0][i] += 1
                if FER[0][i] >= ErrorFrame:
                    break
            BER[0][i] = BER[0][i] / ns / self.Ncarrier / self.qAry
            FER[0][i] = FER[0][i] / ns
            NMSE_dft[0][i] /= ns
            NMSE_idft[0][i] /= ns
            H_NMSE[0][i] /= ns
            rawH_NMSE[0][i] /= ns
        return BER, FER, NMSE_dft, NMSE_idft, H_NMSE, rawH_NMSE

    def SplitFER(self, slice: int = 4):
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
                BitStream = self.Bit_create()
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
            BER[0][i] = BER[0][i] / ns / self.Ncarrier / self.qAry
            FER[0][i] = FER[0][i] / ns
            NMSE_dft[0][i] /= ns
            NMSE_idft[0][i] /= ns
            H_NMSE[0][i] /= ns
            rawH_NMSE[0][i] /= ns
        return BER, FER, NMSE_dft, NMSE_idft, H_NMSE, rawH_NMSE

    def create_Traindata(self, SNR):
        sample = 25000
        DFT_Xtrain = np.zeros((sample, 20), dtype=complex)
        DFT_Ytrain = np.zeros((sample, 128), dtype=complex)
        DFT_W = self.DFTm[0:20]  # 20*128

        IDFT_Xtrain = np.zeros((sample, 128), dtype=complex)
        IDFT_Ytrain = np.zeros((sample, 20), dtype=complex)
        IDFT_W = self.IDFTm[:, 0:20]  # 128*20
        sigma_2 = np.power(10, (SNR / 10))
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

    def pathDetect(self):
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
                BitStream = self.Bit_create()
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

            h_ests[0][i] /= ns
            FER[0][i] /= ns
            NMSE_idfts[0][i] /= ns

        return FER, NMSE_idfts


def save_mat(mat, fname):
    # fpath = os.path.join(NEW_DIR, fname)
    np.save(fname, mat)


def convert_complexToReal_X(X):
    X_real = X.real
    X_img = X.imag
    return np.concatenate([np.concatenate([X_real, -X_img], 1), np.concatenate([X_img, X_real], 1)], 0)


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


params = {
    'Nifft': 128,
    'Ncarrier': 128,
    'qAry': 2,
    'Symbol_len': 128,
    'Symbol_num': 1,
    'ldpc_rate': 0.5,
    'L': 16,
    # 'PathGain': np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    'PathGain': np.linspace(1, 0.1, 16),
    'SNR': np.linspace(-20, 10, 13),
    'ErrorFrame': 20,
    'TestFrame': 5000,
    'Encode_method': None,
    'ncodebooks': 64,
    'ncentroids': 16,
    'matmul_method': METHOD_MITHRAL
}

if __name__ == '__main__':
    _dir = os.path.dirname(os.path.abspath(__file__))
    starttime = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    foutName = _dir + "/results/dft_main_" + starttime + ".txt"
    with open(foutName, "x") as fout:
        fout.write("start at %s\nparams:\n" % starttime)
        pprint(params, fout)
        fout.write("matmul_method: %s\n" % params["matmul_method"])

    myTransceiver = Transceiver(params)
    doPathDetect = False  # 是否是检测径
    doTrain = False  # 是否是生成训练集

    if doTrain:
        myTransceiver.create_Traindata(0)
    elif not doPathDetect:
        BER, FER, NMSE_dft, NMSE_idft, H_NMSE, rawH_NMSE = myTransceiver.FER()
        print("BER", BER)
        print("FER", FER)
        print("NMSE_dft", NMSE_dft)
        print("NMSE_idft", NMSE_idft)
        print("H_NMSE", H_NMSE)
        print("rawH_NMSE", rawH_NMSE)

        stoptime = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        with open(foutName, "a+") as fout:
            fout.write("\nBER:\n")
            np.savetxt(fout, BER, "%.4e")
            fout.write("\nFER:\n")
            np.savetxt(fout, FER, "%.4e")
            fout.write("\nNMSE_dft:\n")
            np.savetxt(fout, NMSE_dft, "%.4e")
            fout.write("\nNMSE_idft:\n")
            np.savetxt(fout, NMSE_idft, "%.4e")
            fout.write("\nH_NMSE:\n")
            np.savetxt(fout, H_NMSE, "%.4e")
            fout.write("\nrawH_NMSE:\n")
            np.savetxt(fout, rawH_NMSE, "%.4e")
            fout.write("stop at %s\n" % stoptime)
    else:  # pathDetect
        params["PathGain"] = np.array(
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        FER, NMSE_idft = myTransceiver.pathDetect()
        print("FER", FER)
        print("NMSE_idft", NMSE_idft)

        stoptime = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        with open(foutName, "a+") as fout:
            fout.write("\nFER:\n")
            np.savetxt(fout, FER, "%.4e")
            fout.write("\nNMSE_idft:\n")
            np.savetxt(fout, NMSE_idft, "%.4e")
            fout.write("stop at %s\n" % stoptime)
