import math
import numpy as np
import math_util
import os
import matmul as mm


class Transceiver:
    def __init__(self, params):
        self.params = params
        self.Nifft = params['Nifft']
        self.Ncarrier = params['Ncarrier']
        self.qAry = params['qAry']
        self.Symbol_len = params['Symbol_len']
        self.Symbol_num = params['Symbol_num']
        self.matmul_method = params['matmul_method']

        self.bitpolit = self.Bit_create()  # 列向量
        self.Xpolit = np.zeros((1, self.Ncarrier), dtype=complex)
        for nf in range(self.Ncarrier):
            self.Xpolit[0, nf] = self.Modulation(
                self.bitpolit[0, 2 * nf:2 * nf + 2])
        self.Xpolit = np.transpose(self.Xpolit)
        self.Create_DFTmatrix()

    def Create_DFTmatrix(self):
        n = np.arange(self.Nifft).reshape(1, self.Nifft)
        k = np.arange(self.Nifft).reshape(1, self.Nifft)
        Wn = np.exp(-1j * 2 * np.pi / self.Nifft)
        nk = np.dot(n.T, k)
        self.DFTm = np.zeros(nk.shape, dtype=complex)
        for i in range(self.DFTm.shape[0]):
            for j in range(self.DFTm.shape[1]):
                self.DFTm[i][j] = np.power(Wn, nk[i][j])
        self.IDFTm = 1/self.Nifft * np.conj(self.DFTm)

    def Bit_create(self):
        # 生成一帧信息比特/导频
        Bitlen = self.qAry * self.Ncarrier * self.Symbol_num
        bitstream = np.random.randint(0, 2, (1, Bitlen))
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

    def Channel_est(self, Ypolit, dft_est, idft_est):
        Hest = Ypolit/self.Xpolit
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
        if self.matmul_method == 0:
            h_est_p = self.IDFT_i(np.transpose(Hest), self.Nifft, est=idft_est)
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
        if self.matmul_method == 1:
            # Exact
            Xk = xp  # TODO
        elif self.matmul_method == 0:
            # Mithral
            xn = convert_complexToReal_X(xn)
            W = convert_complexToReal_W(W)
            Xk_tmp = mm.eval_matmul(est, xn, W)
            Xk = covert_realToComplex_Y(Xk_tmp)
            NMSE_dft = cal_NMSE(Xk_tmp, convert_complexToReal_Y(xp))
        return Xk, NMSE_dft

    def IDFT(self, Xk, N, est=None):
        # 代替ifft
        W = self.IDFTm[:, 0:20]  # 此处已经截取
        NMSE_idft = 0
        xp = np.dot(Xk, W)
        if self.matmul_method == 1:
            # Exact
            xn = xp  # TODO
        elif self.matmul_method == 0:
            # Mithral
            Xk = convert_complexToReal_X(Xk)
            W = convert_complexToReal_W(W)
            xn_tmp = mm.eval_matmul(est, Xk, W)
            xn = covert_realToComplex_Y(xn_tmp)
            NMSE_idft = cal_NMSE(xn_tmp, convert_complexToReal_Y(xp))
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
        ErrorFrame = self.params['ErrorFrame']

        dft_est = None
        idft_est = None
        if self.matmul_method == 0:
            # Mithral
            dft_est = mm.estFactory(
                X_path="DFT_X.npy", W_path="DFT_W.npy", Y_path="DFT_Y.npy", dir="dft")
            idft_est = mm.estFactory(
                X_path="IDFT_X.npy", W_path="IDFT_W.npy", Y_path="IDFT_Y.npy", dir="dft")

        for i, SNR in enumerate(SNRs):
            sigma_2 = np.power(10, (-SNR/10))
            # sigma_2 = 0 # back-to-back
            ns = 0
            print(SNR)
            while FER[0][i] < ErrorFrame:
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
                Ypolit = np.dot(H, self.Xpolit) + np.sqrt(sigma_2/2) * noise
                # Hest_DFT = H # 测试
                Hest_DFT, nmse_dft, nmse_idft, h_nmse = self.Channel_est(
                    Ypolit, dft_est=dft_est, idft_est=idft_est)
                # 更新
                NMSE_dft[0][i] += nmse_dft
                NMSE_idft[0][i] += nmse_idft
                H_NMSE[0][i] += h_nmse

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
        return BER, FER, NMSE_dft, NMSE_idft, H_NMSE

    def create_Traindata(self):
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
            Ypolit = np.dot(H, self.Xpolit) + np.sqrt(sigma_2 / 2) * noise
            Hest = Ypolit / self.Xpolit
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
    'L': 16,
    'PathGain': np.linspace(1, 0.1, 16),
    'SNR': [0, 3, 6, 9, 12, 15, 18, 21],
    'ErrorFrame': 500,
    'Encode_method': None,
    'matmul_method': 0  # 0-Mithral, 1-Exact
}

if __name__ == '__main__':
    myTransceiver = Transceiver(params)
    # myTransceiver.create_Traindata()
    BER, FER, NMSE_dft, NMSE_idft, H_NMSE = myTransceiver.FER()
    print(BER)
    print(FER)
    print(NMSE_dft)
    print(NMSE_idft)
    print(H_NMSE)
