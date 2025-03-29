import pennylane as qml
import torch
import torch.nn as nn
import numpy as np
import math

from new_Preprocessing import padding_conv2d, amp_input_reshape


feature_tolerance = 1e-20
norm_tolerance = 1e-40


# input_feature_new: (batch_size, input_dim, input_row, input_col)
def qcnn(KernelAncilla, input_feature, output_dim, output_size, kernel_size, stride, padding_index, qcnn_layer):

    batch_size = input_feature.shape[0]
    input_dim = input_feature.shape[1]
    input_row = input_feature.shape[2]
    input_col = input_feature.shape[3]
    
    
    output_height_size = output_size[0]
    output_width_size = output_size[1]
    output_total_size = output_height_size*output_width_size
    
    
    kernel_row = kernel_size[0]      
    kernel_col = kernel_size[1]      
    kernel_total_size = kernel_row*kernel_col
    
    
    # 2**conv_size >= output_total_size,  2**6 >= 49
    # 2**kernel_size = kernel_total_size, 2**4 = 16
    # conv_qubit: 6 / kernel_qubit: KernelAncilla + 4 / total_qubit: KernelAncilla + 10 / total_qubit_with_ancilla: 1 + KernelAncilla + 10
    # total_qubit: without counting one ancilla qubit
    conv_qubit = int(np.ceil(math.log2(output_total_size)))     
    kernel_qubit = KernelAncilla + int(np.ceil(math.log2(kernel_total_size)))  
    total_qubit = conv_qubit + kernel_qubit
    

    N = 2**total_qubit

    
    # make sure all the values in "input_feature[input_channel]" are not zero
    input_feature_tmp = torch.reshape(input_feature,(-1, input_row, input_col))
    input_feature_tmp = input_feature_tmp.to(torch.float32)

    
    for ii in range(input_feature_tmp.shape[0]):
        if not input_feature_tmp[ii].any():

            input_feature_tmp[ii] = input_feature_tmp[ii] + feature_tolerance
            

    # padding_input: (batch_size, input_dim, input_row, input_col)
    padding_input = torch.reshape(input_feature_tmp, (batch_size, input_dim, input_row, input_col))
    
    
    # padding_result: (batch_size, input_dim, input_row+padding_index*2, input_col+padding_index*2)
    padding_result = padding_conv2d(padding_input=padding_input, padding_index=padding_index)

    
    # amp_result: (batch_size, input_dim, 2**total_qubit)
    amp_result = amp_input_reshape(KernelAncilla=KernelAncilla,
                                   padding_result=padding_result, output_size=output_size, 
                                   kernel_size=kernel_size, stride=stride)

    
    
    # go through each input_channel in one batch (compute all data in one batch simultaneously) 
    for input_channel in range(input_dim):        
        
        # qcnn_input: (batch_size, 2**total_qubit)
        # qcnn_input_norm: (batch_size,)
        qcnn_input = amp_result[:, input_channel, :]
        qcnn_input_norm = torch.unsqueeze(torch.norm(qcnn_input, dim=1) + norm_tolerance, 1)

        
        # qcnn_orig_input: (batch_size, 2**total_qubit)
        # qcnn_ancilla_input: (batch_size, 1)
        # qcnn_final_input: (batch_size, (2**total_qubit)+1)
        qcnn_orig_input = 1/math.sqrt(2)*(qcnn_input / qcnn_input_norm)
        qcnn_ancilla_input = torch.full((batch_size, 1), 1/math.sqrt(2))
        qcnn_final_input = torch.cat((qcnn_orig_input, qcnn_ancilla_input), 1)
        
      
        
        # layer_output_tmp: (batch_size, (2**total_qubit)*2)
        # layer_shape: (2**total_qubit)*2
        layer_output_tmp = qcnn_layer[input_channel](qcnn_final_input)
        layer_shape = layer_output_tmp.shape[1]

        
        
        # prob_ancilla_0: (batch_size, 2**total_qubit), the first half prob values of layer_output_tmp
        # prob_ancilla_1: (batch_size, 2**total_qubit), the last half prob values of layer_output_tmp
        prob_ancilla_0 = layer_output_tmp[:, 0:int(layer_shape/2)]
        prob_ancilla_1 = layer_output_tmp[:, int(layer_shape/2):layer_shape]
        

        
        # layer_final_output: (batch_size, 2**total_qubit)
        layer_final_output = qcnn_input_norm*math.sqrt(N)*(prob_ancilla_1 - prob_ancilla_0)

        
        
        # output_tmp: (batch_size, 2**total_qubit)
        if input_channel == 0:                    
            output_tmp = layer_final_output    # for the first time
        else: 
            output_tmp = output_tmp + layer_final_output
            

        
        
    # reshape each data in output_tmp based on the (2**conv_qubit) & (2**kernel_qubit)
    # to match the shape of states that we want based on the output_size and kernel_size
    # final_output_tmp_1: (batch_size, 2**conv_qubit, 2**kernel_qubit)
    final_output_tmp_1 = torch.reshape(output_tmp, (batch_size, -1, 2**kernel_qubit))

    
    # transpose each data in final_output_tmp_1
    # final_output_tmp_2: (batch_size, 2**kernel_qubit, 2**conv_qubit)
    final_output_tmp_2 = torch.transpose(final_output_tmp_1, 1, 2)

    
    # make each channel/dim in the final_output_tmp_2 reshape to the output_size (now we have 2**kernel_qubit-channel)
    # final_output_tmp_3: (batch_size, 2**kernel_qubit, output_height_size, output_width_size)
    # only pick the first output_total_size value in the state, others can throw away
    final_output_tmp_3 = torch.reshape(final_output_tmp_2[:,:,0:output_total_size], 
                                       (batch_size, -1, output_height_size, output_width_size))

    
    # only pick the output_channel/output_dim we want
    # final_output: (batch_size, output_dim, output_size[0], output_size[1])
    qcnn_final_output = final_output_tmp_3[:, 0:output_dim]
            
    return qcnn_final_output