# Exercise 3.1: Learning-based Signal Detection for OFDM Systems

This repository provides the starter code for Exercise 3.1. Your task is to use Deep Learning (FC-DNN) to implicitly estimate the channel and recover transmitted bits in an OFDM system, based on the paper by Ye et al. [15]. You will modify network dimensions, modulation schemes, and pilot configurations to evaluate the system's Bit Error Rate (BER).

## Environment Setup (Windows/RTX 4050)
The project is configured for Windows 10/11 with an NVIDIA RTX 4050 GPU for hardware acceleration:
1. **Create the virtual environment with the specific Python version:**
    ```bash
    conda create -n tensorflow python=3.9.19 -y

2. **Activate the environment:**
    ```bash
    conda activate tensorflow

3. **Install CUDA Toolkit via Conda:**
    ```bash
    conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0

4. **Install all required packages via requirements.txt:**
   ```bash
   pip install -r requirements.txt

## Execution Workflow
1. cd to the DNN_Detection 
    ```bash
    cd DNN_Detection
    ```

2. Execute the 
    ```bash
    python Main.py
    ```
    
All Model saved to Model_b, Model_c, Model_d file

The H_dataset is downloaded from the following link: https://github.com/haoyye/OFDM_DNN.
