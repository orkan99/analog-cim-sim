import unittest
import numpy as np
import acs_int
import json
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# indexing instuction (sheet, row - height, column - width)



"""
def unroll(array):
    return np.reshape(array, -1)
    
def extract_ifm_column(ifm, pos_w : int, pos_h : int, kernel_size : tuple, dilation: int = 1):
    if(type(kernel_size) != tuple):
        raise Exception("Kernel must be a tuple!")
    if(kernel_size[1] % 2 == 0 or kernel_size[2] % 2 == 0):
        raise Exception("Not implemented for even kernel size")
    column = ifm[:, pos_h - dilation*(kernel_size[1] - 1) // 2 : pos_h + dilation*(kernel_size[1] - 1) // 2 + 1 : dilation, pos_w - dilation*(kernel_size[2] - 1) // 2: pos_w + dilation*(kernel_size[2] - 1) // 2 + 1 : dilation] # was experimental for dilation, now works good
    column = np.reshape(column, -1)
    if(len(column) != math.prod(kernel_size)):
        raise Exception("Kernel dimensions are different than the lenght of the extracted part of IMF (one column). Something is wrong!")
    return column

def extract_ifm_matrix(ifm, kernel_size: tuple, stride : int = 1, padding : int = 0, dilation : int = 1):
    if(stride < 1 or padding < 0 or dilation < 1):
        raise Exception("Something is wrong with passed parameters!")
    out_w = int(np.floor( (np.shape(ifm)[2] + 2 * padding - dilation * (kernel_size[2] -1) - 1) / stride + 1 )) # based on Pytorch formulas for Conv2d from https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d
    out_h = int(np.floor( (np.shape(ifm)[1] + 2 * padding - dilation * (kernel_size[1] -1) - 1) / stride + 1 ))
    ifm_matrix = np.zeros((math.prod(kernel_size) , out_h*out_w))
    ifm = np.pad(ifm, ((0,0), (padding, padding), (padding, padding)))
    counter = 0
    for i in range(out_w):
        for j in range(out_h):
            ifm_matrix[:, counter] = extract_ifm_column(ifm, pos_w = i*stride + dilation*(kernel_size[1]-1)//2, pos_h = j*stride + dilation*(kernel_size[2] - 1)//2, kernel_size=kernel_size, dilation = dilation) # works good for everything
            counter += 1
            
    return ifm_matrix, out_h, out_w

def extract_kernel_matrix(kernels):
    if(len(np.shape(kernels)) != 4):
        raise Exception("Something with dimensions is wrong, the input should be a 4D array which represents a list of 3D kernels!")
    return np.reshape(kernels, (np.shape(kernels)[0], -1))

def calculate_ofm_matrix(ifm_matrix, kernel_matrix, cfg_file_path = "analog-cim-sim/cpp/test/lib/configs/digital/I_DIFF_W_DIFF_1XB.json"):
    with open(cfg_file_path, "r") as file:
            cfg = json.load(file)
    ofm_matrix = np.zeros((np.shape(kernel_matrix)[0], np.shape(ifm_matrix)[1]), dtype=np.int32)
    m_crossbar = cfg['M']
    n_crossbar = cfg['N']
    vg_num = int(np.ceil(np.shape(kernel_matrix)[1]/n_crossbar))
    hg_num = int(np.ceil(np.shape(kernel_matrix)[0]/m_crossbar))
    acs_int.set_config(os.path.abspath(cfg_file_path))
    for  i in range(hg_num):
        for j in range(vg_num):
            cut = kernel_matrix[i*m_crossbar: (i + 1)*m_crossbar, j*n_crossbar: (j + 1)*n_crossbar]
            m_matrix, n_matrix = np.shape(cut)
            acs_int.cpy(np.reshape(cut, -1), m_matrix, n_matrix)
            for k in range(np.shape(ifm_matrix)[1]):
                result = np.zeros(m_matrix,dtype=np.int32)
                acs_int.mvm(result, ifm_matrix[j*n_crossbar : (j+1)*n_crossbar, k], np.reshape(cut, -1), m_matrix, n_matrix)
                ofm_matrix[i*m_crossbar : (i+1)*m_crossbar, k] += result
    return ofm_matrix

def shape_ofm(ofm_matrix, out_w, out_h):
    out_c = np.shape(ofm_matrix)[0]
    out = np.ones((out_c, out_h, out_w))*-1
    for i in range(out_c):
        out[i][:][:] = np.reshape(ofm_matrix[i], (out_h, out_w), order='F')
    return out

def acs_conv2(ifm, kernels, padding = 0, stride = 1, dilation = 1):    
    kernel_matrix = extract_kernel_matrix(kernels = kernels)
    ifm_matrix, out_h, out_w = extract_ifm_matrix(ifm, np.shape(kernels)[-3:], padding = padding, stride = stride, dilation = dilation)
    ofm_matrix = calculate_ofm_matrix(ifm_matrix = ifm_matrix, kernel_matrix= kernel_matrix)
    ofm = shape_ofm(ofm_matrix = ofm_matrix, out_h = out_h, out_w = out_w)
    return ofm

def acs_fc(ifm, w_matrix, bias = None, cfg_file_path = "analog-cim-sim/cpp/test/lib/configs/digital/I_DIFF_W_DIFF_1XB.json"):
    with open(cfg_file_path, "r") as file:
            cfg = json.load(file)
    if(bias is None):
        bias = np.zeros((1,np.shape(w_matrix)[0]))
    else:
        bias = np.reshape(bias, (1,-1))
        if np.shape(bias)[-1] != np.shape(w_matrix)[0]:
            if(len(np.shape())):
                raise Exception("Bias dimension is not compatible with weights matrix and desired number of outputs!")
    ofm_t = np.zeros((np.shape(w_matrix)[0], np.shape(ifm)[0]), dtype=np.int32)
    ifm_t = np.transpose(ifm)
    m_crossbar = cfg['M']
    n_crossbar = cfg['N']
    vg_num = int(np.ceil(np.shape(w_matrix)[1]/n_crossbar))
    hg_num = int(np.ceil(np.shape(w_matrix)[0]/m_crossbar))
    acs_int.set_config(os.path.abspath(cfg_file_path))
    for  i in range(hg_num):
        for j in range(vg_num):
            cut = w_matrix[i*m_crossbar: (i + 1)*m_crossbar, j*n_crossbar: (j + 1)*n_crossbar]
            m_matrix, n_matrix = np.shape(cut)
            acs_int.cpy(np.reshape(cut, -1), m_matrix, n_matrix)
            for k in range(np.shape(ifm_t)[1]):
                result = np.zeros(m_matrix, dtype=np.int32)
                acs_int.mvm(result, ifm_t[j*n_crossbar : (j+1)*n_crossbar, k], np.reshape(cut, -1), m_matrix, n_matrix)
                ofm_t[i*m_crossbar : (i+1)*m_crossbar, k] += result
    return np.transpose(ofm_t) + bias
"""
def unroll(array):
    return np.reshape(array, -1)
    
def extract_ifm_column(ifm, pos_w : int, pos_h : int, kernel_size : tuple, dilation: int = 1):
    #if(type(kernel_size) != tuple):
      #  raise Exception("Kernel must be a tuple!")
    if(kernel_size[1] % 2 == 0 or kernel_size[2] % 2 == 0):
        raise Exception("Not implemented for even kernel size")
    #column = ifm[:, pos_h - dilation*(kernel_size[1] - 1) // 2 : pos_h + dilation*(kernel_size[1] - 1) // 2 + 1 : dilation, pos_w - dilation*(kernel_size[2] - 1) // 2: pos_w + dilation*(kernel_size[2] - 1) // 2 + 1 : dilation] # was experimental for dilation, now works good
    if(type(ifm) == torch.Tensor or type(ifm) == np.ndarray):
        column = ifm[:, slice(pos_h - dilation*(kernel_size[1] - 1) // 2 , pos_h + dilation*(kernel_size[1] - 1) // 2 + 1 , dilation), slice(pos_w - dilation*(kernel_size[2] - 1) // 2, pos_w + dilation*(kernel_size[2] - 1) // 2 + 1 , dilation)] # the approach with slices works both for np arrays and tensors
    else: 
        column = ifm[0][:, slice(pos_h - dilation*(kernel_size[1] - 1) // 2 , pos_h + dilation*(kernel_size[1] - 1) // 2 + 1 , dilation), slice(pos_w - dilation*(kernel_size[2] - 1) // 2, pos_w + dilation*(kernel_size[2] - 1) // 2 + 1 , dilation)] # the approach with slices works both for np arrays and tensors
    column = np.reshape(column, -1)
    if(len(column) != math.prod(kernel_size)):
        raise Exception("Kernel dimensions are different than the lenght of the extracted part of IMF (one column). Something is wrong!")
    return column

def extract_ifm_matrix(ifm, kernel_size: tuple, stride : int = 1, padding : int = 0, dilation : int = 1):
    if(stride < 1 or padding < 0 or dilation < 1):
        raise Exception("Something is wrong with passed parameters!")
    out_w = int(np.floor( (np.shape(ifm)[2] + 2 * padding - dilation * (kernel_size[2] -1) - 1) / stride + 1 )) # based on Pytorch formulas for Conv2d from https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d
    out_h = int(np.floor( (np.shape(ifm)[1] + 2 * padding - dilation * (kernel_size[1] -1) - 1) / stride + 1 ))
    ifm_matrix = np.zeros((math.prod(kernel_size) , out_h*out_w))
    if(type(ifm) == np.ndarray):
        ifm = np.pad(ifm, ((0,0), (padding, padding), (padding, padding)))
    else:
        ifm = F.pad(input=ifm, pad=(padding, padding, padding, padding), mode='constant', value=0)
    counter = 0
    for i in range(out_w):
        for j in range(out_h):
            ifm_matrix[:, counter] = extract_ifm_column(ifm, pos_w = i*stride + dilation*(kernel_size[1]-1)//2, pos_h = j*stride + dilation*(kernel_size[2] - 1)//2, kernel_size=kernel_size, dilation = dilation) # works good for everything
            counter += 1
            
    return ifm_matrix, out_h, out_w

def extract_kernel_matrix(kernels):
    if(len(np.shape(kernels)) != 4):
        raise Exception("Something with dimensions is wrong, the input should be a 4D array which represents a list of 3D kernels!")
    return np.reshape(kernels, (np.shape(kernels)[0], -1))

def calculate_ofm_matrix(ifm_matrix, kernel_matrix, cfg_file_path = "analog-cim-sim/cpp/test/lib/configs/digital/I_DIFF_W_DIFF_1XB.json"):
    with open(cfg_file_path, "r") as file:
            cfg = json.load(file)
    ofm_matrix = np.zeros((np.shape(kernel_matrix)[0], np.shape(ifm_matrix)[1]), dtype=np.int32)
    m_crossbar = cfg['M']
    n_crossbar = cfg['N']
    vg_num = int(np.ceil(np.shape(kernel_matrix)[1]/n_crossbar))
    hg_num = int(np.ceil(np.shape(kernel_matrix)[0]/m_crossbar))
    acs_int.set_config(os.path.abspath(cfg_file_path))
    for  i in range(hg_num):
        for j in range(vg_num):
            cut = np.array(kernel_matrix[i*m_crossbar: (i + 1)*m_crossbar, j*n_crossbar: (j + 1)*n_crossbar], dtype=np.int32)
            m_matrix, n_matrix = np.shape(cut)
            acs_int.cpy(np.reshape(cut, -1), m_matrix, n_matrix)
            for k in range(np.shape(ifm_matrix)[1]):
                result = np.zeros(m_matrix,dtype=np.int32)
                acs_int.mvm(result, ifm_matrix[j*n_crossbar : (j+1)*n_crossbar, k], np.reshape(cut, -1), m_matrix, n_matrix)
                res = np.matmul(cut, ifm_matrix[j*n_crossbar : (j+1)*n_crossbar, k])
                if(not ((res - result) == 0).all()):
                    print("A problem occured! ")
                ofm_matrix[i*m_crossbar : (i+1)*m_crossbar, k] += result
    return ofm_matrix

def shape_ofm(ofm_matrix, out_w, out_h):
    out_c = np.shape(ofm_matrix)[0]
    out = np.ones((out_c, out_h, out_w))*-1
    for i in range(out_c):
        out[i][:][:] = np.reshape(ofm_matrix[i], (out_h, out_w), order='F')
    return out

def acs_conv2(ifm, kernels, padding = 0, stride = 1, dilation = 1, bias = None):    
    kernel_matrix = extract_kernel_matrix(kernels = kernels)
    
    ofm_final = [] # number of images in a batch
    for i in range(ifm.size()[0]):
        if(type(ifm) == torch.Tensor or type(ifm) == np.ndarray):
            ifm_matrix, out_h, out_w = extract_ifm_matrix(ifm = ifm[i], kernel_size=np.shape(kernels)[-3:], padding = padding, stride = stride, dilation = dilation)
        else:
            ifm_matrix, out_h, out_w = extract_ifm_matrix(ifm = ifm[0][i], kernel_size=np.shape(kernels)[-3:], padding = padding, stride = stride, dilation = dilation)
        ofm_matrix = calculate_ofm_matrix(ifm_matrix = ifm_matrix, kernel_matrix= kernel_matrix)
        ofm_final.append(torch.from_numpy(shape_ofm(ofm_matrix = ofm_matrix, out_h = out_h, out_w = out_w)).float())
        

    ofm_final = torch.stack(ofm_final)
    if(bias is not None):
        ofm_final += bias.view([1,kernels.size()[0],1,1])
    return ofm_final

def acs_fc(ifm, w_matrix, bias = None, cfg_file_path = "analog-cim-sim/cpp/test/lib/configs/digital/I_DIFF_W_DIFF_1XB.json"):
    with open(cfg_file_path, "r") as file:
            cfg = json.load(file)
    if(bias is None):
        bias = np.zeros((1,np.shape(w_matrix)[0]))
    else:
        bias = np.reshape(bias, (1,-1))
        if np.shape(bias)[-1] != np.shape(w_matrix)[0]:
            if(len(np.shape())):
                raise Exception("Bias dimension is not compatible with weights matrix and desired number of outputs!")
    ofm_t = np.zeros((np.shape(w_matrix)[0], np.shape(ifm)[0]), dtype=np.int32)
    ifm_t = np.transpose(ifm)
    m_crossbar = cfg['M']
    n_crossbar = cfg['N']
    vg_num = int(np.ceil(np.shape(w_matrix)[1]/n_crossbar))
    hg_num = int(np.ceil(np.shape(w_matrix)[0]/m_crossbar))
    acs_int.set_config(os.path.abspath(cfg_file_path))
    for  i in range(hg_num):
        for j in range(vg_num):
            cut = w_matrix[i*m_crossbar: (i + 1)*m_crossbar, j*n_crossbar: (j + 1)*n_crossbar]
            m_matrix, n_matrix = np.shape(cut)
            acs_int.cpy(np.reshape(cut, -1), m_matrix, n_matrix)
            for k in range(np.shape(ifm_t)[1]):
                result = np.zeros(m_matrix, dtype=np.int32)
                acs_int.mvm(result, ifm_t[j*n_crossbar : (j+1)*n_crossbar, k], np.reshape(cut, -1), m_matrix, n_matrix)
                ofm_t[i*m_crossbar : (i+1)*m_crossbar, k] += result
    return np.transpose(ofm_t) + bias


class Test_Conv2d(unittest.TestCase):

    def test_one_kernel_with_one_dimension(self):
        # Just one krernel with one dimension (1, 3, 3) - less than m_matrix and n_matrix - proven to work
        
        x = np.arange(20, dtype=np.int32)
        x = np.reshape(x, (1,5,4))

        kernels = np.array([np.ones((1,3,3), dtype=np.int32)*i for i in range(1,2)])
        kernel_matrix = extract_kernel_matrix(kernels = kernels)
        padding = [1,2,3]
        stride = [1,2,3]
        for p in padding:
            for s in stride:
                    ifm_matrix, out_h, out_w = extract_ifm_matrix(x, np.shape(kernels)[-3:], padding = p, stride = s, dilation = 1)
                    ofm_matrix = calculate_ofm_matrix(ifm_matrix = ifm_matrix, kernel_matrix= kernel_matrix)
                    ofm = shape_ofm(ofm_matrix=ofm_matrix, out_h=out_h, out_w=out_w)

                    torch_conv2 = nn.functional.conv2d(torch.from_numpy(x), torch.from_numpy(kernels), padding=p, stride=s, dilation= 1).numpy()
                    np.testing.assert_array_equal(ofm, torch_conv2)
    
    def test_one_kernel_with_multiple_dimensions(self):
        # Just one kernel with multiple dimensions (number, 3, 3) - less than m_matrix or n_matrix - proven to work

        x = np.arange(60, dtype=np.int32)
        x = np.reshape(x, (3,5,4))

        kernels = np.array([np.ones((3,3,3), dtype=np.int32)*i for i in range(1,2)])
        kernel_matrix = extract_kernel_matrix(kernels = kernels)

        ifm_matrix, out_h, out_w = extract_ifm_matrix(x, np.shape(kernels)[-3:], padding = 0, stride = 1, dilation = 1)
        ofm_matrix = calculate_ofm_matrix(ifm_matrix = ifm_matrix, kernel_matrix= kernel_matrix)
        ofm = shape_ofm(ofm_matrix=ofm_matrix, out_h=out_h, out_w=out_w)

        torch_conv2 = nn.functional.conv2d(torch.from_numpy(x), torch.from_numpy(kernels)).numpy()
        np.testing.assert_array_equal(ofm, torch_conv2)

    def test_multiple_kernels_with_one_dimension(self):
        # Just 2 kernels with one dimension (1, 3, 3) - less than m_matrix or n_matrix - proven to work
        
        x = np.arange(20, dtype=np.int32)
        x = np.reshape(x, (1,5,4))

        kernels = np.array([np.ones((1,3,3), dtype=np.int32)*i for i in range(1,3)])
        kernel_matrix = extract_kernel_matrix(kernels = kernels)

        ifm_matrix, out_h, out_w = extract_ifm_matrix(x, np.shape(kernels)[-3:], padding = 0, stride = 1, dilation = 1)
        ofm_matrix = calculate_ofm_matrix(ifm_matrix = ifm_matrix, kernel_matrix= kernel_matrix)
        ofm = shape_ofm(ofm_matrix=ofm_matrix, out_h=out_h, out_w=out_w)

        torch_conv2 = nn.functional.conv2d(torch.from_numpy(x), torch.from_numpy(kernels)).numpy()
        np.testing.assert_array_equal(ofm, torch_conv2)
    
    def test_multiple_kernels_with_multiple_dimensions(self):
        # Just 20 kernels with multiple dimensions (3, 3, 3) - less than m_matrix or n_matrix - proven to work
        
        x = np.arange(60, dtype=np.int32)
        x = np.reshape(x, (3,5,4))

        kernels = np.array([np.ones((3,3,3), dtype=np.int32)*i for i in range(1,21)])
        kernel_matrix = extract_kernel_matrix(kernels = kernels)

        ifm_matrix, out_h, out_w = extract_ifm_matrix(x, np.shape(kernels)[-3:], padding = 0, stride = 1, dilation = 1)
        ofm_matrix = calculate_ofm_matrix(ifm_matrix = ifm_matrix, kernel_matrix= kernel_matrix)
        ofm = shape_ofm(ofm_matrix=ofm_matrix, out_h=out_h, out_w=out_w)

        torch_conv2 = nn.functional.conv2d(torch.from_numpy(x), torch.from_numpy(kernels)).numpy()
        np.testing.assert_array_equal(ofm, torch_conv2)

    
    def test_one_kernel_with_many_dimensions(self):
        # Just one kernel with multiple dimensions (8, 3, 3) - less than m_matrix but more than n_matrix - proven to work
                
        x = np.arange(160, dtype=np.int32)
        x = np.reshape(x, (8,5,4))

        kernels = np.array([np.ones((8,3,3), dtype=np.int32)*i for i in range(1,2)])
        kernel_matrix = extract_kernel_matrix(kernels = kernels)

        ifm_matrix, out_h, out_w = extract_ifm_matrix(x, np.shape(kernels)[-3:], padding = 0, stride = 1, dilation = 1)
        ofm_matrix = calculate_ofm_matrix(ifm_matrix = ifm_matrix, kernel_matrix= kernel_matrix)
        ofm = shape_ofm(ofm_matrix=ofm_matrix, out_h=out_h, out_w=out_w)

        torch_conv2 = nn.functional.conv2d(torch.from_numpy(x), torch.from_numpy(kernels)).numpy()
        np.testing.assert_array_equal(ofm, torch_conv2)

    def test_many_kernels_with_one_dimension(self):
        # Just many kernels with one dimension (1, 3, 3) - more than m_matrix but less than n_matrix - proven to work
                
        x = np.arange(20, dtype=np.int32)
        x = np.reshape(x, (1,5,4))

        kernels = np.array([np.ones((1,3,3), dtype=np.int32)*i for i in range(1,70)])
        kernel_matrix = extract_kernel_matrix(kernels = kernels)

        ifm_matrix, out_h, out_w = extract_ifm_matrix(x, np.shape(kernels)[-3:], padding = 0, stride = 1, dilation = 1)
        ofm_matrix = calculate_ofm_matrix(ifm_matrix = ifm_matrix, kernel_matrix= kernel_matrix)
        ofm = shape_ofm(ofm_matrix=ofm_matrix, out_h=out_h, out_w=out_w)

        torch_conv2 = nn.functional.conv2d(torch.from_numpy(x), torch.from_numpy(kernels)).numpy()
        np.testing.assert_array_equal(ofm, torch_conv2)

    def test_many_kernels_with_many_dimensions(self):
        # Just many kernels with multiple dimensions (8, 3, 3) - more than m_matrix and more than n_matrix - proven to work
                
        x = np.arange(160, dtype=np.int32)
        x = np.reshape(x, (8,5,4))

        kernels = np.array([np.ones((8,3,3), dtype=np.int32)*i for i in range(1,70)])
        kernel_matrix = extract_kernel_matrix(kernels = kernels)

        padding = [1,2,3,4]
        stride = [1,2,3]
        for p in padding:
            for s in stride:
                    ifm_matrix, out_h, out_w = extract_ifm_matrix(x, np.shape(kernels)[-3:], padding = p, stride = s, dilation = 1)
                    ofm_matrix = calculate_ofm_matrix(ifm_matrix = ifm_matrix, kernel_matrix= kernel_matrix)
                    ofm = shape_ofm(ofm_matrix=ofm_matrix, out_h=out_h, out_w=out_w)

                    torch_conv2 = nn.functional.conv2d(torch.from_numpy(x), torch.from_numpy(kernels), padding=p, stride=s, dilation= 1).numpy()
                    np.testing.assert_array_equal(ofm, torch_conv2)

                                

class Test_FC(unittest.TestCase):

    def test_one_small_input_no_bias(self):
        # One input smaller than n_matrix with no bias
        torch.manual_seed(0)
        input_f = 5
        output_f = 2
        batch_size = 1
        x = np.arange(input_f*batch_size, dtype=np.float32)
        x = np.reshape(x, (batch_size, input_f))
        w_matrix = torch.arange(input_f*output_f, dtype=torch.float32)
        w_matrix = torch.reshape(w_matrix,(output_f,input_f))
        
        out = acs_fc(ifm = np.array(x, dtype=np.int32), w_matrix = np.array(w_matrix, dtype=np.int32))
        
        linear = nn.Linear(input_f, output_f, bias=False)
        linear.weight.data = w_matrix
        out_torch = linear(torch.from_numpy(x)).detach().numpy()
        np.testing.assert_array_equal(out, out_torch) 
    
    def test_one_small_input_with_bias(self):
        # One input smaller than n_matrix with bias
        torch.manual_seed(0)
        input_f = 5
        output_f = 2
        batch_size = 1
        x = np.arange(input_f*batch_size, dtype=np.float32)
        x = np.reshape(x, (batch_size, input_f))
        w_matrix = torch.arange(input_f*output_f, dtype=torch.float32)
        w_matrix = torch.reshape(w_matrix,(output_f,input_f))
        bias = torch.arange(output_f, dtype=torch.float32)*3
        
        out = acs_fc(ifm = np.array(x, dtype=np.int32), w_matrix = np.array(w_matrix, dtype=np.int32), bias=np.array(bias, dtype=np.int32))
        
        linear = nn.Linear(input_f, output_f, bias=True)
        linear.weight.data = w_matrix
        linear.bias.data = bias
        out_torch = linear(torch.from_numpy(x)).detach().numpy()
        np.testing.assert_array_equal(out, out_torch) 
    
    def test_one_big_input_with_bias(self):
        # One input larger than n_matrix with bias
        torch.manual_seed(0)
        input_f = 72
        output_f = 2
        batch_size = 1
        x = np.arange(input_f*batch_size, dtype=np.float32)
        x = np.reshape(x, (batch_size, input_f))
        w_matrix = torch.arange(input_f*output_f, dtype=torch.float32)
        w_matrix = torch.reshape(w_matrix,(output_f,input_f))
        bias = torch.arange(output_f, dtype=torch.float32)*3
        
        out = acs_fc(ifm = np.array(x, dtype=np.int32), w_matrix = np.array(w_matrix, dtype=np.int32), bias=np.array(bias, dtype=np.int32))


        
        linear = nn.Linear(input_f, output_f, bias=True)
        linear.weight.data = w_matrix
        linear.bias.data = bias
        out_torch = linear(torch.from_numpy(x)).detach().numpy()
        np.testing.assert_array_equal(out, out_torch) 

    def test_many_small_input_with_bias(self):
        # One input larger than n_matrix with bias
        torch.manual_seed(0)
        input_f = 5
        output_f = 22
        batch_size = 104
        x = np.arange(input_f*batch_size, dtype=np.float32)
        x = np.reshape(x, (batch_size, input_f))
        w_matrix = torch.arange(input_f*output_f, dtype=torch.float32)
        w_matrix = torch.reshape(w_matrix,(output_f,input_f))
        bias = torch.arange(output_f, dtype=torch.float32)*3
        
        out = acs_fc(ifm = np.array(x, dtype=np.int32), w_matrix = np.array(w_matrix, dtype=np.int32), bias=np.array(bias, dtype=np.int32))
        
        linear = nn.Linear(input_f, output_f, bias=True)
        linear.weight.data = w_matrix
        linear.bias.data = bias
        out_torch = linear(torch.from_numpy(x)).detach().numpy()
        np.testing.assert_array_equal(out, out_torch) 

    def test_many_big_inputs_with_bias(self):
        #Many (more than m_matrix) inputs larger than n_matrix with bias
        torch.manual_seed(0)
        input_f = 256
        output_f = 156
        batch_size = 1115 
        x = np.ones(input_f*batch_size, dtype=np.float32)
        x = np.reshape(x, (batch_size, input_f))
        w_matrix = torch.ones(input_f*output_f, dtype=torch.float32)
        w_matrix = torch.reshape(w_matrix,(output_f,input_f))
        bias = torch.arange(output_f, dtype=torch.float32)*3
        
        out = acs_fc(ifm = np.array(x, dtype=np.int64), w_matrix = np.array(w_matrix, dtype=np.int64), bias=np.array(bias, dtype=np.int64))
        
        linear = nn.Linear(input_f, output_f, bias=True)
        linear.weight.data = w_matrix
        linear.bias.data = bias
        out_torch = linear(torch.from_numpy(x)).detach().numpy()
        np.testing.assert_array_equal(out, out_torch) 

if __name__ == "__main__":
    unittest.main()
