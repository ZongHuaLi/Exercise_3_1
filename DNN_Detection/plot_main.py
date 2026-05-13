import os
from plottestchar import plottestchar
import numpy as np
import matplotlib.pyplot as plt

# 不同小題用不同配置，全部由 main.py 控制
class sysconfig_b (object):
    Pilots = 8        # number of pilots
    with_CP_flag = True 
    SNR = 20
    Clipping = False
    Train_set_path = '../H_dataset/'
    Test_set_path = '../H_dataset/'
    Model_path = '../Models_b/'
    pred_range = np.arange(16,32)
    learning_rate = 0.001
    learning_rate_decrease_step = 2000    

    # 增加不同小題須修改的配置 
    mu = 2
    n_input = 256
    n_hidden_1 = 500
    n_hidden_2 = 250
    n_hidden_3 = 120
    n_output = 16

class sysconfig_c (object):
    Pilots = 8        # number of pilots
    with_CP_flag = True 
    SNR = 20
    Clipping = False
    Train_set_path = '../H_dataset/'
    Test_set_path = '../H_dataset/'
    Model_path = '../Models_c/'
    pred_range = np.arange(48,96) #64QAM
    learning_rate = 0.001
    learning_rate_decrease_step = 2000    

    # 增加不同小題須修改的配置 
    mu = 6 # 64QAM
    n_input = 256
    n_hidden_1 = 500
    n_hidden_2 = 250
    n_hidden_3 = 120
    n_output = 48 # 64QAM

class sysconfig_d (object):
    Pilots = 8        # number of pilots
    with_CP_flag = True 
    SNR = 20
    Clipping = False
    Train_set_path = '../H_dataset/'
    Test_set_path = '../H_dataset/'
    Model_path = '../Models_d/'
    pred_range = np.arange(0,128)
    learning_rate = 0.001
    learning_rate_decrease_step = 2000    

    # 增加不同小題須修改的配置 
    mu = 2
    n_input = 256
    n_hidden_1 = 1000
    n_hidden_2 = 500
    n_hidden_3 = 240
    n_output = 128


def main():
    SNR_list = [5, 10, 15, 20, 25] 
    
    # 字典來儲存所有的 BER
    # results[task][pilot] = [ber_snr5, ber_snr10, ...]
    results = {
        'b': {8: [], 16: [], 64: []}, 
        'c': {8: []},                 
        'd': {8: []}                  
    }

    print("\n==========(b) Task B: 不同導頻數量比較 ==========\n")
    pilot_list_b = [8, 16, 64]
    for p in pilot_list_b:
        for snr in SNR_list:
            print(f"\n----------(b) Run Pilots: {p}, SNR: {snr} dB----------")
            config = sysconfig_b()
            config.Pilots = p
            config.SNR = snr
            config.model_dir = config.Model_path + 'SNR_' + str(snr) + '/'

            
            ber = plottestchar(config)  # 接收 Test.py 回傳的 BER
            results['b'][p].append(ber) # 存入清單

    print("\n==========(c) Task C: 64-QAM 測試 ==========\n")
    for snr in SNR_list:
        p = 8 # 維持其他條件不變，Pilot 用 8
        print(f"\n----------(c) Run Pilots: {p}, SNR: {snr} dB----------")
        config = sysconfig_c()
        config.Pilots = p
        config.SNR = snr
        config.model_dir = config.Model_path + 'SNR_' + str(snr) + '/'

        
        ber = plottestchar(config)
        results['c'][p].append(ber)

    print("\n==========(d) Task D: 單一大型 DNN 測試 ==========\n")
    for snr in SNR_list:
        p = 8 
        print(f"\n----------(d) Run Pilots: {p}, SNR: {snr} dB----------")
        config = sysconfig_d()
        config.Pilots = p
        config.SNR = snr
        config.model_dir = config.Model_path + 'SNR_' + str(snr) + '/'

        ber = plottestchar(config)
        results['d'][p].append(ber)


    print("\nStarting to plot...")

    # 繪製 Task (b) 比較圖: 不同 Pilot 數量
    plt.figure(figsize=(8, 6))
    plt.semilogy(SNR_list, results['b'][64], marker='o', linestyle='-', label='64 Pilots (QPSK, FC-DNN)')
    plt.semilogy(SNR_list, results['b'][16], marker='s', linestyle='--', label='16 Pilots (QPSK, FC-DNN)')
    plt.semilogy(SNR_list, results['b'][8],  marker='^', linestyle='-.', label='8 Pilots (QPSK, FC-DNN)')
    plt.grid(True, which="both", ls="--")
    plt.xlabel('SNR (dB)')
    plt.ylabel('Bit Error Rate (BER)')
    plt.title('Task (b): Impact of Pilot Numbers')
    plt.legend()
    plt.savefig('Figure_Task_B.png')
    plt.close()

    # 繪製 Task (c) 比較圖: QPSK vs 64-QAM
    plt.figure(figsize=(8, 6))
    plt.semilogy(SNR_list, results['b'][8], marker='o', linestyle='-', label='QPSK (8 Pilots)')
    plt.semilogy(SNR_list, results['c'][8], marker='s', linestyle='-', label='64-QAM (8 Pilots)')
    plt.grid(True, which="both", ls="--")
    plt.xlabel('SNR (dB)')
    plt.ylabel('Bit Error Rate (BER)')
    plt.title('Task (c): QPSK vs 64-QAM Modulation')
    plt.legend()
    plt.savefig('Figure_Task_C.png')
    plt.close()

    # 繪製 Task (d) 比較圖: 8個小DNN vs 1個大DNN
    plt.figure(figsize=(8, 6))
    plt.semilogy(SNR_list, results['b'][8], marker='o', linestyle='-', label='8 Identical FC-DNNs (Output=16)')
    plt.semilogy(SNR_list, results['d'][8], marker='^', linestyle='-', label='1 Single Large FC-DNN (Output=128)')
    plt.grid(True, which="both", ls="--")
    plt.xlabel('SNR (dB)')
    plt.ylabel('Bit Error Rate (BER)')
    plt.title('Task (d): Parallel DNNs vs Single Large DNN')
    plt.legend()
    plt.savefig('Figure_Task_D.png')
    plt.close()

    print("Output Figure_Task_B.png, Figure_Task_C.png, Figure_Task_D.png")

if __name__ == "__main__":
    main()