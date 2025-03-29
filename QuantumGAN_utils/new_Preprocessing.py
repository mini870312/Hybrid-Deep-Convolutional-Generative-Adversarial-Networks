import torch
import torch.nn as nn
import numpy as np
import math


# padding_input: (batch_size, input_channel_size, input_row, input_col)
def padding_conv2d(padding_input, padding_index):
    
    batch_size = padding_input.shape[0]
    channel_size = padding_input.shape[1]
    
    output_row = padding_input.shape[2] + padding_index*2        
    output_col = padding_input.shape[3] + padding_index*2 

    # padding_result: (batch_size, input_channel_size, padding_result_row, padding_result_col)
    padding_result = torch.zeros(batch_size, channel_size, output_row, output_col)
    padding_result[:, :, padding_index:output_row-padding_index, padding_index:output_col-padding_index] = padding_input

    return padding_result




# padding_result: (batch_size, input_channel_size, padding_result_row, padding_result_col)
def amp_input_reshape(KernelAncilla, padding_result, output_size, kernel_size, stride):

    
    batch_size = padding_result.shape[0] 
    channel_size = padding_result.shape[1]  
    
    padding_result_row = padding_result.shape[2]      
    padding_result_col = padding_result.shape[3] 

    output_total_size = output_size[0]*output_size[1]

    kernel_row = kernel_size[0]      
    kernel_col = kernel_size[1]      
    kernel_total_size = kernel_row*kernel_col

    # example
    # 2**conv_size >= output_total_size,  2**6 >= 49
    # 2**kernel_size = kernel_total_size, 2**4 = 16
    # conv_qubit: 6 / kernel_qubit: KernelAncilla + 4 / total_qubit: KernelAncilla + 10 / total_qubit_with_ancilla: 1 + KernelAncilla + 10
    conv_qubit = int(np.ceil(math.log2(output_total_size)))     
    kernel_qubit = KernelAncilla + int(np.ceil(math.log2(kernel_total_size)))  
    total_qubit = conv_qubit + kernel_qubit
    
    amp_result = torch.zeros(batch_size, channel_size, 2**total_qubit)

    count = 0
    for ii in range(kernel_row, padding_result_row+1, stride):
        for jj in range(kernel_col, padding_result_col+1, stride):
                
            # kernel_scan: (batch_size, input_channel_size, kernel_row, kernel_col)
            kernel_scan = padding_result[:,:,ii-kernel_row:ii,jj-kernel_col:jj]

            
            # kernel_scan_reshape: (batch_size, input_channel_size, kernel_total_size)          
            kernel_scan_reshape = torch.reshape(kernel_scan, (batch_size, channel_size, -1))
            
            
            # amp_input_tmp: (batch_size, input_channel_size, 2**kernel_qubit)
            amp_input_tmp = torch.zeros(batch_size, channel_size, 2**kernel_qubit)


            amp_input_tmp[:,:,0:kernel_total_size] = kernel_scan_reshape
            

            # amp_result: (batch_size, input_channel_size, 2**total_qubit)
            amp_result[:,:,count*(2**kernel_qubit):(1+count)*(2**kernel_qubit)] = amp_input_tmp

        
            # count: kernel scanning times, related to output_total_size
            count = count + 1

    
    return amp_result


    