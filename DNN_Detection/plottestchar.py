from __future__ import division
import numpy as np
import scipy.interpolate 
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import math
import os
from utils import *


def plottestchar(config): # 加入 config参数
        tf.reset_default_graph()  # 清除前一個 Task 殘留的 Graph
        # OFDM 參數同 Train.py
        K = 64
        CP = K//4
        P = config.Pilots # number of pilot carriers per OFDM block
        allCarriers = np.arange(K)  # indices of all subcarriers ([0, 1, ... K-1])
        mu = config.mu # 改由main.py的config控制
        CP_flag = config.with_CP_flag
        if P<K:
            pilotCarriers = allCarriers[::K//P] # Pilots is every (K/P)th carrier.
            dataCarriers = np.delete(allCarriers, pilotCarriers)
            
        else:   # K = P
            pilotCarriers = allCarriers
            dataCarriers = []


        payloadBits_per_OFDM = K*mu  
        
        SNRdb = config.SNR  # signal to noise-ratio in dB at the receiver 
        Clipping_Flag = config.Clipping 
        
        #Pilot_file_name = 'Pilot_'+str(P)
        Pilot_file_name = 'Pilot_' + str(P) + '_mu_' + str(mu) + '.txt'
        print(Pilot_file_name)
        if os.path.isfile(Pilot_file_name):
            print ('Load Training Pilots txt')
            bits_pilot = np.loadtxt(Pilot_file_name, delimiter=',')
            pilotValue = Modulation(bits_pilot, mu) # 必須轉換成 pilotValue
        else:
            print('Error: No Pilot txt file')
            return


        # Network Parameters
        n_hidden_1 = config.n_hidden_1 # 1st layer num features 改由main.py的config控制
        n_hidden_2 = config.n_hidden_2 # 2nd layer num features 改由main.py的config控制
        n_hidden_3 = config.n_hidden_3 # 3rd layer num features 改由main.py的config控制
        n_input = config.n_input # MNIST data input (img shape: 28*28) 改由main.py的config控制
        n_output = config.n_output # Output features 改由main.py的config控制
        # tf Graph input (only pictures)
        X = tf.placeholder("float", [None, n_input])
        #Y = tf.placeholder("float", [None, K*mu])
        Y = tf.placeholder("float", [None, n_output])
        def encoder(x):
            weights = {                    
                'encoder_h1': tf.Variable(tf.truncated_normal([n_input, n_hidden_1],stddev=0.1)),
                'encoder_h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2],stddev=0.1)),
                'encoder_h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3],stddev=0.1)),
                'encoder_h4': tf.Variable(tf.truncated_normal([n_hidden_3, n_output],stddev=0.1)),            
            }
            biases = {            
                'encoder_b1': tf.Variable(tf.truncated_normal([n_hidden_1],stddev=0.1)),
                'encoder_b2': tf.Variable(tf.truncated_normal([n_hidden_2],stddev=0.1)),
                'encoder_b3': tf.Variable(tf.truncated_normal([n_hidden_3],stddev=0.1)),
                'encoder_b4': tf.Variable(tf.truncated_normal([n_output],stddev=0.1)),          
            
            }
        
            # Encoder Hidden layer with sigmoid activation #1
            #layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
            layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
            layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))
            layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, weights['encoder_h3']), biases['encoder_b3']))
            layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, weights['encoder_h4']), biases['encoder_b4']))
            return layer_4
        # Building the decoder

        #encoder_op = encoder(X)

        #for network_idx in range(0, int(K*mu/n_output)):
        #    y_pred_cur = encoder(X)
        #    if network_idx == 0:
        #        y_pred = y_pred_cur
        #    else:
        #        y_pred = tf.concat((y_pred, y_pred_cur), axis=1)            
        # Prediction
        y_pred = encoder(X)
        # Targets (Labels) are the input data.
        y_true = Y

        # Define loss and optimizer, minimize the squared error
        cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
        #cost = tf.reduce_mean(tf.pow(y_true - y_pred, 1))
        #cost = tf.reduce_mean(tf.abs(y_true-y_pred))
        learning_rate = tf.placeholder(tf.float32, shape=[])
        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

        # Initializing the variables
        init = tf.global_variables_initializer()
        
        # Generating Detection 
        #code = BinaryLinearBlockCode(parityCheckMatrix='./test/data/BCH_63_36_5_strip.alist')
        #code = PolarCode(6, SNR=4, mu = 16, rate = 0.5)
        #decoders = [IterativeDecoder(code, minSum=True, iterations=50, reencodeOrder=-1, reencodeRange=0.1)]        

        # Start Training
        config_GPU = tf.ConfigProto()
        config_GPU.gpu_options.allow_growth = True
        # The H information set
        test_idx_low = 1
        test_idx_high = 80      
        '''
        H_folder = '../H_dataset/'
        test_idx_low = 301
        test_idx_high = 400 
        '''
        H_folder = config.Test_set_path # 使用 config.Test_set_path
        channel_response_set_test = []
        for test_idx in range(test_idx_low,test_idx_high):
            H_file = H_folder + str(test_idx) + '.txt'
            with open(H_file) as f:
                for line in f:
                    numbers_str = line.split()
                    numbers_float = [float(x) for x in numbers_str]
                    h_response = np.asarray(numbers_float[0:int(len(numbers_float)/2)])+1j*np.asarray(numbers_float[int(len(numbers_float)/2):len(numbers_float)])
                    channel_response_set_test.append(h_response)




        print ('length of testing channel response', len(channel_response_set_test))



        saver = tf.train.Saver()
        
        init = tf.global_variables_initializer()
        tf_config = tf.ConfigProto() # 修正錯誤：將 tf.ConfigProto() 改為 tf_config，避免覆蓋傳入的 config
        tf_config.gpu_options.allow_growth = True # 修正錯誤

        with tf.Session(config=tf_config) as sess: # 修正錯誤
            sess.run(init)            
            #saving_name = config.model_name
            #saver.restore(sess, saving_name)
            
            # 自動抓取資料夾內最新儲存的模型
            latest_model = tf.train.latest_checkpoint(config.model_dir)
            if latest_model is None:
                print(f"Error: 在 {config.model_dir} 找不到任何模型檔！")
                return
            
            print(f"成功載入模型: {latest_model}")
            saver.restore(sess, latest_model)             
            input_samples_test = []
            input_labels_test = []
            test_number = 100000        
            for i in range(0, test_number):
                # 修正正確的 payloadBits 算法
                payloadBits_per_OFDM = K * mu 
                bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM, )) 
                
                channel_response = channel_response_set_test[np.random.randint(0,len(channel_response_set_test))]
                
                #補齊 12 個參數，且使用 SNRdb 而非 config.SNRdb
                signal_output, para = ofdm_simulate(bits, channel_response, SNRdb, mu, CP_flag, K, P, CP, pilotValue, pilotCarriers, dataCarriers, Clipping_Flag)
                
                input_labels_test.append(bits[config.pred_range])
                input_samples_test.append(signal_output)

                #bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM, )) 
                #signal_train, signal_output, para = ofdm_simulate(bits) 
                #codeword = code.encode(bits)    
                #signal_train, signal_output, para = ofdm_simulate(codeword) 
                #channel_response= channel_response_set_test[np.random.randint(0,len(channel_response_set_test))]
                #signal_output, para = ofdm_simulate_single(bits,channel_response)
                #signal_output, para = ofdm_simulate(bits,channel_response,config.SNRdb)
                #input_labels_test.append(codeword)
                #input_labels_test.append(bits[config.pred_range])
                #input_samples_test.append(np.concatenate((signal_train,signal_output)))
                #input_samples_test.append(signal_output)
                        
            batch_x = np.asarray(input_samples_test)
            batch_y = np.asarray(input_labels_test)
            encode_decode = sess.run(y_pred, feed_dict = {X:batch_x})
            mean_error = tf.reduce_mean(abs(y_pred - batch_y))                
            BER = 1-tf.reduce_mean(tf.reduce_mean(tf.to_float(tf.equal(tf.sign(y_pred-0.5), tf.cast(tf.sign(batch_y-0.5),tf.float32))),1))
                        
            # 取出 BER
            final_ber = BER.eval({X:batch_x})
            
            print("OFDM Detection QAM output number is", n_output, "SNR = ", SNRdb, "Num Pilot", P, "prediction and the mean error on test set are:", mean_error.eval({X:batch_x}), final_ber)
            
            # 回傳 BER
            return final_ber


