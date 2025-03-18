import unittest
import numpy as np
import acs_int
import json
import os
import math
import torch
import torch.nn as nn
# indexing instuction (sheet, row - height, column - width)

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
        raise Exception("SOmehting with dimensions is wrong, the input should be a 4D array which represents a list of 3D kernels!")
    return np.reshape(kernels, (np.shape(kernels)[0], -1))

def calculate_ofm_matrix(ifm_matrix, kernel_matrix, cfg_file_path = "cpp/test/lib/configs/digital/I_DIFF_W_DIFF_1XB.json"):
    """def test_digital_I_DIFF_W_DIFF_1XB(self):
        m_matrix = 3
        n_matrix = 2
        mat = np.array([100, -32, 1, 0, 12, 1], dtype=np.int32)
        vec = np.array([-120, 55], dtype=np.int32)
        res = np.array([1, 1, -1], dtype=np.int32)

        
        acs_int.cpy(mat, m_matrix, n_matrix)
        acs_int.mvm(res, vec, mat, m_matrix, n_matrix)
        np.testing.assert_array_equal(res, np.array([-13759, -119, -1386], dtype=np.int32))"""
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
                ofm_matrix[i*n_crossbar : (i+1)*m_crossbar, k] += result
    return ofm_matrix

def shape_ofm(ofm_matrix, out_w, out_h):
    out_c = np.shape(ofm_matrix)[0]
    out = np.ones((out_c, out_h, out_w))*-1
    """counter = 0
    for i in range(out_c):
        for j in range(out_w):
            for k in range(out_h):
                out[i][k][j] = ofm_matrix[counter]"""
    for i in range(out_c):
        out[i][:][:] = np.reshape(ofm_matrix[i], (out_h, out_w), order='F')
    return out
    
"""
x = np.arange(120, dtype=np.int32)
x = np.reshape(x, (6,5,4))

# making 4 kernels of dimensions (6,3,3) will produce the output 4*3*4
kernels = np.array([np.ones((6,3,3), dtype=np.int32)*i for i in range(1,2)])
kernel_matrix = extract_kernel_matrix(kernels = kernels)

ifm_matrix, out_h, out_w = extract_ifm_matrix(x, np.shape(kernels)[-3:], padding = 0, stride = 1, dilation = 1)
ofm_matrix = calculate_ofm_matrix(ifm_matrix = ifm_matrix, kernel_matrix= kernel_matrix)
ofm = shape_ofm(ofm_matrix=ofm_matrix, out_h=out_h, out_w=out_w)

#torch_conv2 = nn.functional.conv2d(torch.from_numpy(x), torch.from_numpy(kernels)).numpy()

print(ofm)
"""

"""

x = np.arange(160, dtype=np.int32)
x = np.reshape(x, (8,5,4))

kernels = np.array([np.ones((8,3,3), dtype=np.int32)*i for i in range(1,2)])
kernel_matrix = extract_kernel_matrix(kernels = kernels)

ifm_matrix, out_h, out_w = extract_ifm_matrix(x, np.shape(kernels)[-3:], padding = 0, stride = 1, dilation = 1)
ofm_matrix = calculate_ofm_matrix(ifm_matrix = ifm_matrix, kernel_matrix= kernel_matrix)
ofm = shape_ofm(ofm_matrix=ofm_matrix, out_h=out_h, out_w=out_w)

print(ofm) 

#torch_conv2 = nn.functional.conv2d(torch.from_numpy(x), torch.from_numpy(kernels)).numpy()


#print(torch_conv2)

print(out_w, out_h)


print("dokle vise")

"""
class Test_Andrija(unittest.TestCase):

    def test_one_kernel_with_one_dimension(self):
        # Just one krernel with one dimension (1, 3, 3) - less than m_matrix and n_matrix - proven to work
        
        x = np.arange(20, dtype=np.int32)
        x = np.reshape(x, (1,5,4))

        kernels = np.array([np.ones((1,3,3), dtype=np.int32)*i for i in range(1,2)])
        kernel_matrix = extract_kernel_matrix(kernels = kernels)

        ifm_matrix, out_h, out_w = extract_ifm_matrix(x, np.shape(kernels)[-3:], padding = 0, stride = 1, dilation = 1)
        ofm_matrix = calculate_ofm_matrix(ifm_matrix = ifm_matrix, kernel_matrix= kernel_matrix)
        ofm = shape_ofm(ofm_matrix=ofm_matrix, out_h=out_h, out_w=out_w)

        #print(ofm)
        torch_conv2 = nn.functional.conv2d(torch.from_numpy(x), torch.from_numpy(kernels)).numpy()
        np.testing.assert_array_equal(ofm, torch_conv2)
        #print(torch_conv2)
    
    def test_one_kernel_with_multiple_dimensions(self):
        # Just one kernel with multiple dimensions (number, 3, 3) - less than m_matrix or n_matrix - proven to work

        x = np.arange(60, dtype=np.int32)
        x = np.reshape(x, (3,5,4))

        kernels = np.array([np.ones((3,3,3), dtype=np.int32)*i for i in range(1,2)])
        kernel_matrix = extract_kernel_matrix(kernels = kernels)

        ifm_matrix, out_h, out_w = extract_ifm_matrix(x, np.shape(kernels)[-3:], padding = 0, stride = 1, dilation = 1)
        ofm_matrix = calculate_ofm_matrix(ifm_matrix = ifm_matrix, kernel_matrix= kernel_matrix)
        ofm = shape_ofm(ofm_matrix=ofm_matrix, out_h=out_h, out_w=out_w)

        #print(ofm)

        torch_conv2 = nn.functional.conv2d(torch.from_numpy(x), torch.from_numpy(kernels)).numpy()
        np.testing.assert_array_equal(ofm, torch_conv2)

        #print(torch_conv2)

    def test_multiple_kernels_with_one_dimension(self):
        # Just 2 kernels with one dimension (1, 3, 3) - less than m_matrix or n_matrix - proven to work
        
        x = np.arange(20, dtype=np.int32)
        x = np.reshape(x, (1,5,4))

        kernels = np.array([np.ones((1,3,3), dtype=np.int32)*i for i in range(1,3)])
        kernel_matrix = extract_kernel_matrix(kernels = kernels)

        ifm_matrix, out_h, out_w = extract_ifm_matrix(x, np.shape(kernels)[-3:], padding = 0, stride = 1, dilation = 1)
        ofm_matrix = calculate_ofm_matrix(ifm_matrix = ifm_matrix, kernel_matrix= kernel_matrix)
        ofm = shape_ofm(ofm_matrix=ofm_matrix, out_h=out_h, out_w=out_w)

        #print(ofm)
        torch_conv2 = nn.functional.conv2d(torch.from_numpy(x), torch.from_numpy(kernels)).numpy()
        np.testing.assert_array_equal(ofm, torch_conv2)


        #print(torch_conv2)
    
    def test_one_kernel_with_many_dimensions(self):
        # Just one kernel with multiple dimensions (8, 3, 3) - less than m_matrix but more than n_matrix - proven to work
                
        x = np.arange(160, dtype=np.int32)
        x = np.reshape(x, (8,5,4))

        kernels = np.array([np.ones((8,3,3), dtype=np.int32)*i for i in range(1,2)])
        kernel_matrix = extract_kernel_matrix(kernels = kernels)

        ifm_matrix, out_h, out_w = extract_ifm_matrix(x, np.shape(kernels)[-3:], padding = 0, stride = 1, dilation = 1)
        ofm_matrix = calculate_ofm_matrix(ifm_matrix = ifm_matrix, kernel_matrix= kernel_matrix)
        ofm = shape_ofm(ofm_matrix=ofm_matrix, out_h=out_h, out_w=out_w)

        #print(ofm) 

        torch_conv2 = nn.functional.conv2d(torch.from_numpy(x), torch.from_numpy(kernels)).numpy()
        np.testing.assert_array_equal(ofm, torch_conv2)

        #print(torch_conv2)
                
                
if __name__ == "__main__":
    unittest.main()

