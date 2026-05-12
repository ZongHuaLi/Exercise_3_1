import os
from Train import train
from Test import test
import numpy as np

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
    #SNR_list = [5] # for test
    pilot_list = [8, 16, 64]
    #pilot_list = [8] # for test

    # b, c, d 分別執行不同配置的訓練和測試
    print("\n==========(b)==========")
    for p in pilot_list:
        for snr in SNR_list:
            print(f"\n----------(b)Run Pilots: {p}, SNR: {snr} dB----------")
            config = sysconfig_b()
            config.Pilots = p
            config.SNR = snr
            #config.model_name = config.Model_path + 'SNR_' + str(snr) + '/DetectionModel_SNR_' + str(snr) + '_Pilot_' + str(p) + '_epoch_25'
            config.model_dir = config.Model_path + 'SNR_' + str(snr) + '/'

            print("\ntraining...")
            train(config)
            
            print("\ntesting...")
            test(config)

    print("\n==========(c)==========")
    for p in pilot_list:
        for snr in SNR_list:
            print(f"\n----------(c)Run Pilots: {p}, SNR: {snr} dB----------")
            config = sysconfig_c()
            config.Pilots = p
            config.SNR = snr
            #config.model_name = config.Model_path + 'SNR_' + str(snr) + '/DetectionModel_SNR_' + str(snr) + '_Pilot_' + str(p) + '_epoch_25'
            config.model_dir = config.Model_path + 'SNR_' + str(snr) + '/'

            print("\ntraining...")
            train(config)
            
            print("\ntesting...")
            test(config)

    print("\n==========(d)==========")
    for p in pilot_list:
        for snr in SNR_list:
            print(f"\n----------(d)Run Pilots: {p}, SNR: {snr} dB----------")
            config = sysconfig_d()
            config.Pilots = p
            config.SNR = snr
            #config.model_name = config.Model_path + 'SNR_' + str(snr) + '/DetectionModel_SNR_' + str(snr) + '_Pilot_' + str(p) + '_epoch_25'
            config.model_dir = config.Model_path + 'SNR_' + str(snr) + '/'

            print("\ntraining...")
            train(config)
            
            print("\ntesting...")
            test(config)

main()

