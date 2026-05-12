from __future__ import division
import numpy as np
# import scipy.interpolate 
# import tensorflow as tf
import math
import os
def print_something():
    print ('utils.py has been loaded perfectly')





def Clipping (x, CL):
    sigma = np.sqrt(np.mean(np.square(np.abs(x))))
    CL = CL*sigma
    x_clipped = x  
    clipped_idx = abs(x_clipped) > CL
    x_clipped[clipped_idx] = np.divide((x_clipped[clipped_idx]*CL),abs(x_clipped[clipped_idx]))
    return x_clipped

def PAPR(x):
    Power = np.abs(x)**2
    PeakP = np.max(Power)
    AvgP = np.mean(Power)
    PAPR_dB = 10*np.log10(PeakP/AvgP)
    return PAPR_dB

def Modulation(bits,mu):                                        
    bit_r = bits.reshape((int(len(bits)/mu), mu))                  
    #return (2*bit_r[:,0]-1)+1j*(2*bit_r[:,1]-1)                                    # This is just for QAM modulation
    # 根據 mu 的值來決定使用哪種調變方式，假設 mu=2 是 QPSK，mu=6 是 64-QAM
    if mu == 2: # QPSK
        return (2*bit_r[:,0]-1) + 1j*(2*bit_r[:,1]-1)
    
    elif mu == 6: # 64-QAM (Gray Mapping 簡化版)
        # 實部由前 3 bits 決定，虛部由後 3 bits 決定
        # 這裡提供一個標準的 PAM 映射來組成 64-QAM
        def bits_to_pam(b):
            # b 是 shape (N, 3) 的矩陣
            return (1-2*b[:,0]) * (4 - (1-2*b[:,1]) * (2 - (1-2*b[:,2])))
            
        real_part = bits_to_pam(bit_r[:, 0:3])
        imag_part = bits_to_pam(bit_r[:, 3:6])
        # 正規化平均功率 (64-QAM 的平均功率為 42)
        return (real_part + 1j*imag_part) / np.sqrt(42)
        
    else:
        raise ValueError("Currently only supports mu=2 (QPSK) and mu=6 (64-QAM)")


def IDFT(OFDM_data):
    return np.fft.ifft(OFDM_data)

def addCP(OFDM_time, CP, CP_flag, mu, K):
    
    if CP_flag == False:
        # add noise CP
        bits_noise = np.random.binomial(n=1, p=0.5, size=(K*mu, ))
        codeword_noise = Modulation(bits_noise, mu)
        OFDM_data_nosie = codeword_noise
        OFDM_time_noise = np.fft.ifft(OFDM_data_nosie)
        cp = OFDM_time_noise[-CP:]
    else:
        cp = OFDM_time[-CP:]               # take the last CP samples ...
    #cp = OFDM_time[-CP:] 
    return np.hstack([cp, OFDM_time])  # ... and add them to the beginning

def channel(signal,channelResponse,SNRdb):
    convolved = np.convolve(signal, channelResponse)
    signal_power = np.mean(abs(convolved**2))
    sigma2 = signal_power * 10**(-SNRdb/10)  
    noise = np.sqrt(sigma2/2) * (np.random.randn(*convolved.shape)+1j*np.random.randn(*convolved.shape))
    return convolved + noise

def removeCP(signal, CP, K):
    return signal[CP:(CP+K)]

def DFT(OFDM_RX):
    return np.fft.fft(OFDM_RX)


def equalize(OFDM_demod, Hest):
    return OFDM_demod / Hest

def get_payload(equalized):
    return equalized[dataCarriers]


def PS(bits):
    return bits.reshape((-1,))

'''
def Modulation(bits, mu):


    bit_r = bits.reshape((int(len(bits)/mu), mu))                  
    return (2*bit_r[:,0]-1)+1j*(2*bit_r[:,1]-1)                                    # This is just for QAM modulation


def IDFT(OFDM_data):
    return np.fft.ifft(OFDM_data)

def addCP(OFDM_time):
    cp = OFDM_time[-CP:]               # take the last CP samples ...
    return np.hstack([cp, OFDM_time])  # ... and add them to the beginning

def channel(signal,channelResponse,SNRdb):
    convolved = np.convolve(signal, channelResponse)
    signal_power = np.mean(abs(convolved**2))
    sigma2 = signal_power * 10**(-SNRdb/10)  
    noise = np.sqrt(sigma2/2) * (np.random.randn(*convolved.shape)+1j*np.random.randn(*convolved.shape))
    return convolved + noise

def removeCP(signal):
    return signal[CP:(CP+K)]

def DFT(OFDM_RX):
    return np.fft.fft(OFDM_RX)


def equalize(OFDM_demod, Hest):
    return OFDM_demod / Hest

def get_payload(equalized):
    return equalized[dataCarriers]


def PS(bits):
    return bits.reshape((-1,))
'''

def ofdm_simulate(codeword, channelResponse,SNRdb, mu, CP_flag, K, P, CP, pilotValue,pilotCarriers, dataCarriers,Clipping_Flag):  
    payloadBits_per_OFDM = mu*len(dataCarriers)
    # --- training inputs ----
    if P < K:
        bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM, ))    
        QAM = Modulation(bits,mu)
        OFDM_data = np.zeros(K, dtype=complex)
        #OFDM_data[pilotCarriers] = pilotValue  # allocate the pilot subcarriers 
        OFDM_data[pilotCarriers] = pilotValue[pilotCarriers]  # 加上 [pilotCarriers]，確保 pilotValue 的長度與 pilotCarriers 一致
        OFDM_data[dataCarriers] = QAM
    else:
        OFDM_data = pilotValue
    
    OFDM_time = IDFT(OFDM_data)
    OFDM_withCP = addCP(OFDM_time, CP, CP_flag, mu, K)
    #OFDM_withCP = addCP(OFDM_time)
    OFDM_TX = OFDM_withCP
    if Clipping_Flag:
        OFDM_TX = Clipping(OFDM_TX,CR)                            # add clipping 
    OFDM_RX = channel(OFDM_TX, channelResponse,SNRdb)
    OFDM_RX_noCP = removeCP(OFDM_RX, CP,K)
    #OFDM_RX_noCP = removeCP(OFDM_RX)
    # ----- target inputs ---
    symbol = np.zeros(K, dtype=complex)
    codeword_qam = Modulation(codeword,mu)
    if len(codeword_qam) != K:
        print ('length of code word is not equal to K, error !!')
    symbol = codeword_qam
    OFDM_data_codeword = symbol
    OFDM_time_codeword = np.fft.ifft(OFDM_data_codeword)
    OFDM_withCP_cordword = addCP(OFDM_time_codeword, CP, CP_flag, mu, K)
    #OFDM_withCP_cordword = addCP(OFDM_time_codeword)
    if Clipping_Flag:
        OFDM_withCP_cordword = Clipping(OFDM_withCP_cordword,CR) # add clipping 
    OFDM_RX_codeword = channel(OFDM_withCP_cordword, channelResponse,SNRdb)
    OFDM_RX_noCP_codeword = removeCP(OFDM_RX_codeword,CP,K)
    #OFDM_RX_noCP_codeword = removeCP(OFDM_RX_codeword)
    return np.concatenate((np.concatenate((np.real(OFDM_RX_noCP),np.imag(OFDM_RX_noCP))), np.concatenate((np.real(OFDM_RX_noCP_codeword),np.imag(OFDM_RX_noCP_codeword))))), abs(channelResponse) #sparse_mask



'''

def ofdm_simulate(codeword, channelResponse,SNRdb,mu, CP_flag, K, P, CP, pilotValue,pilotCarriers, dataCarriers,Clipping_Flag):       
    payloadBits_per_OFDM = mu*len(dataCarriers)

    # --- training inputs ----
    if P < K:
        bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM, ))  
        QAM = Modulation(bits,mu)
        OFDM_data = np.zeros(K, dtype=complex)
        OFDM_data[pilotCarriers] = pilotValue  # allocate the pilot subcarriers 
        OFDM_data[dataCarriers] = QAM
    else:
        OFDM_data = pilotValue
    OFDM_time = IDFT(OFDM_data)
    OFDM_withCP = addCP(OFDM_time)
    OFDM_TX = OFDM_withCP
    OFDM_RX = channel(OFDM_TX, channelResponse,SNRdb)
    OFDM_RX_noCP = removeCP(OFDM_RX)

    # ----- target inputs ---
    symbol = np.zeros(K, dtype=complex)
    codeword_qam = Modulation(codeword,mu)
    symbol[np.arange(K)] = codeword_qam
    OFDM_data_codeword = symbol
    OFDM_time_codeword = np.fft.ifft(OFDM_data_codeword)
    OFDM_withCP_cordword = addCP(OFDM_time_codeword)
    OFDM_RX_codeword = channel(OFDM_withCP_cordword, channelResponse,SNRdb)
    OFDM_RX_noCP_codeword = removeCP(OFDM_RX_codeword)
    return np.concatenate((np.concatenate((np.real(OFDM_RX_noCP),np.imag(OFDM_RX_noCP))), np.concatenate((np.real(OFDM_RX_noCP_codeword),np.imag(OFDM_RX_noCP_codeword))))), abs(channelResponse) 
'''









'''


def ofdm_simulate(codeword, channelResponse,SNRdb, mu, CP_flag, K, P, CP, pilotValue, pilotCarriers, dataCarriers,Clipping_Flag):       
    OFDM_data = np.zeros(K, dtype=complex)
    allCarriers = np.arange(K)
    OFDM_data[allCarriers] = pilotValue
    OFDM_time = IDFT(OFDM_data)
    OFDM_withCP = addCP(OFDM_time, CP_flag, mu, K)
    OFDM_TX = OFDM_withCP
    OFDM_RX = channel(OFDM_TX, channelResponse,SNRdb)
    OFDM_RX_noCP = removeCP(OFDM_RX,CP,K)

    # ----- target inputs ---
    symbol = np.zeros(K, dtype=complex)
    codeword_qam = Modulation(codeword,mu)
    symbol[np.arange(K)] = codeword_qam
    OFDM_data_codeword = symbol
    OFDM_time_codeword = np.fft.ifft(OFDM_data_codeword)
    OFDM_withCP_cordword = addCP(OFDM_time_codeword, CP_flag, mu, K)
    OFDM_RX_codeword = channel(OFDM_withCP_cordword, channelResponse,SNRdb)
    OFDM_RX_noCP_codeword = removeCP(OFDM_RX_codeword,CP,K)
    return np.concatenate((np.concatenate((np.real(OFDM_RX_noCP),np.imag(OFDM_RX_noCP))), np.concatenate((np.real(OFDM_RX_noCP_codeword),np.imag(OFDM_RX_noCP_codeword))))), abs(channelResponse) 


'''


