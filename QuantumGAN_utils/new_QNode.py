import pennylane as qml
import numpy as np
import math
import torch



def create_qcnn_node(KernelAncilla, output_size, kernel_size, layer_num):
    
    output_total_size = output_size[0]*output_size[1]
    kernel_total_size = kernel_size[0]*kernel_size[1]
    
    conv_qubit = int(np.ceil(math.log2(output_total_size)))     
    kernel_qubit = KernelAncilla + int(np.ceil(math.log2(kernel_total_size))) 
    # add "KernelAncilla" more kernel qubit for expanding the dimension of conv kernel
    
    
    total_qubit = conv_qubit + kernel_qubit 
    total_qubit_with_anc = 1 + total_qubit

 

    
    
    def trainable_kernel(trainable_weights):
        
        kernel_qubit_start = total_qubit_with_anc - kernel_qubit
        
        for ii in range(layer_num):
            for jj in range(kernel_qubit):
                qml.RY(trainable_weights[ii][jj], wires=kernel_qubit_start+jj)

            for jj in range(kernel_qubit-1):
                qml.CNOT(wires=[kernel_qubit_start+jj, kernel_qubit_start+jj+1])


    def superpostion():
        
        for ii in range(1,total_qubit_with_anc):
            qml.Hadamard(wires=ii)
    
    

    
    dev = qml.device("default.qubit", wires=total_qubit_with_anc)

    @qml.qnode(dev)
    def qcnn_node(inputs, weights):
        
        qml.AmplitudeEmbedding(features=inputs, wires=range(total_qubit_with_anc), pad_with=0.)

        qml.ctrl(trainable_kernel, control=0, control_values=(0))(trainable_weights=weights)

        qml.ctrl(superpostion, control=0, control_values=(1))()

        qml.RY(math.pi/2, wires=0)
       
        return qml.probs(wires=range(total_qubit_with_anc))
            
    return qcnn_node