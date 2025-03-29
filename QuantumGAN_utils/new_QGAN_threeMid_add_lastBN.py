import pennylane as qml
import torch
import torch.nn as nn
import numpy as np
import math


from new_QNode import create_qcnn_node
from new_QCNN import qcnn





# class QuantumDiscriminator(nn.Module):
#     def __init__(self, KernelAncilla, layer_num, 
#                  stage_1_dim, stage_1_input_size, stage_1_kernel_size, stage_1_stride, stage_1_padding, 
#                  stage_2_dim, stage_2_input_size, stage_2_kernel_size, stage_2_stride, stage_2_padding,
#                  stage_3_dim, stage_3_input_size, stage_3_kernel_size, stage_3_stride, stage_3_padding,
#                  final_dim, final_size):  
        
#         super().__init__()

#         self.KernelAncilla = KernelAncilla
        
#         self.BatchNorm2d_stage_2_dim = nn.BatchNorm2d(stage_2_dim)
#         self.BatchNorm2d_stage_3_dim = nn.BatchNorm2d(stage_3_dim)

#         self.LeakyReLU = nn.LeakyReLU(True)
#         self.Sigmoid = nn.Sigmoid()
        

        
#         self.stage_1_dim = stage_1_dim
#         self.stage_1_input_size = stage_1_input_size
#         self.stage_1_kernel_size = stage_1_kernel_size
#         self.stage_1_stride = stage_1_stride
#         self.stage_1_padding = stage_1_padding
        
#         self.stage_2_dim = stage_2_dim
#         self.stage_2_input_size = stage_2_input_size
#         self.stage_2_kernel_size = stage_2_kernel_size
#         self.stage_2_stride = stage_2_stride
#         self.stage_2_padding = stage_2_padding
                 
#         self.stage_3_dim = stage_3_dim
#         self.stage_3_input_size = stage_3_input_size
#         self.stage_3_kernel_size = stage_3_kernel_size
#         self.stage_3_stride = stage_3_stride
#         self.stage_3_padding = stage_3_padding    
        
            
#         self.final_dim = final_dim
#         self.final_size = final_size

        
#         self.kernel_qubit_1 = KernelAncilla + int(np.ceil(math.log2(stage_1_kernel_size[0]*stage_1_kernel_size[1]))) 
#         self.kernel_qubit_2 = KernelAncilla + int(np.ceil(math.log2(stage_2_kernel_size[0]*stage_2_kernel_size[1]))) 
#         self.kernel_qubit_3 = KernelAncilla + int(np.ceil(math.log2(stage_3_kernel_size[0]*stage_3_kernel_size[1]))) 
        
#         # -------------------------------- stage 1: 1*7*7 --> 4*5*5 --------------------------------
        
        
#         self.stage_1_qcnn_node = create_qcnn_node(KernelAncilla=KernelAncilla, 
#                                                   output_size=stage_2_input_size, 
#                                                   kernel_size=stage_1_kernel_size, 
#                                                   layer_num=layer_num)
#         self.tmp_layer_1 = []
#         for ii in range(stage_1_dim):
#             self.tmp_layer_1.append(qml.qnn.TorchLayer(self.stage_1_qcnn_node, {"weights": (layer_num, self.kernel_qubit_1)}))
        
#         self.qdis_stage_1 = nn.Sequential(*self.tmp_layer_1)

        
#         # -------------------------------- stage 2: 4*5*5 --> 8*3*3 --------------------------------

#         self.stage_2_qcnn_node = create_qcnn_node(KernelAncilla=KernelAncilla,
#                                                   output_size=stage_3_input_size, 
#                                                   kernel_size=stage_2_kernel_size, 
#                                                   layer_num=layer_num)
#         self.tmp_layer_2 = []
#         for ii in range(stage_2_dim):
#             self.tmp_layer_2.append(qml.qnn.TorchLayer(self.stage_2_qcnn_node, {"weights": (layer_num, self.kernel_qubit_2)}))
        
#         self.qdis_stage_2 = nn.Sequential(*self.tmp_layer_2)
        
        
#         # -------------------------------- stage 3: 8*3*3 --> 1*1*1 --------------------------------

#         self.stage_3_qcnn_node = create_qcnn_node(KernelAncilla=KernelAncilla,
#                                                   output_size=final_size, 
#                                                   kernel_size=stage_3_kernel_size, 
#                                                   layer_num=layer_num)
#         self.tmp_layer_3 = []
#         for ii in range(stage_3_dim):
#             self.tmp_layer_3.append(qml.qnn.TorchLayer(self.stage_3_qcnn_node, {"weights": (layer_num, self.kernel_qubit_3)}))
        
#         self.qdis_stage_3 = nn.Sequential(*self.tmp_layer_3)

    
    
#     def forward(self, x):
                
#         # do qcnn and activation function on all the data in this batch
#         # x: (batch_size, input_dim, input_row, input_col)

#         # -------------------------------- stage 1: 1*7*7 --> 4*5*5 --------------------------------

#         # qcnn_stage_1_output: (batch_size, stage_2_dim, stage_2_input_size[0], stage_2_input_size[1])
#         qcnn_stage_1_output = qcnn(KernelAncilla=self.KernelAncilla, input_feature=x, 
#                                    output_dim=self.stage_2_dim, output_size=self.stage_2_input_size,
#                                    kernel_size=self.stage_1_kernel_size, stride=self.stage_1_stride, 
#                                    padding_index=self.stage_1_padding, qcnn_layer=self.qdis_stage_1)

#         # qcnn_stage_2_input: (batch_size, stage_2_dim, stage_2_input_size[0], stage_2_input_size[1])
#         qcnn_stage_2_input = self.LeakyReLU(self.BatchNorm2d_stage_2_dim(qcnn_stage_1_output))

#         # -------------------------------- stage 2: 4*5*5 --> 8*3*3 --------------------------------

        
#         # qcnn_stage_2_output: (batch_size, stage_3_dim, stage_3_input_size[0], stage_3_input_size[1])
#         qcnn_stage_2_output = qcnn(KernelAncilla=self.KernelAncilla, input_feature=qcnn_stage_2_input, 
#                                    output_dim=self.stage_3_dim, output_size=self.stage_3_input_size, 
#                                    kernel_size=self.stage_2_kernel_size, stride=self.stage_2_stride, 
#                                    padding_index=self.stage_2_padding, qcnn_layer=self.qdis_stage_2)
        
#         # qcnn_stage_3_input: (batch_size, stage_3_dim, stage_3_input_size[0], stage_3_input_size[1])
#         qcnn_stage_3_input = self.LeakyReLU(self.BatchNorm2d_stage_3_dim(qcnn_stage_2_output))


#         # -------------------------------- stage 3: 8*3*3 --> 1*1*1 --------------------------------

        
#         # qcnn_stage_3_output: (batch_size, final_dim, final_size[0], final_size[1])
#         qcnn_stage_3_output = qcnn(KernelAncilla=self.KernelAncilla, input_feature=qcnn_stage_3_input, 
#                                    output_dim=self.final_dim, output_size=self.final_size, 
#                                    kernel_size=self.stage_3_kernel_size, stride=self.stage_3_stride, 
#                                    padding_index=self.stage_3_padding, qcnn_layer=self.qdis_stage_3)

        
#         # final_output: (batch_size, final_dim, final_size[0], final_size[1])
#         final_output = self.Sigmoid(qcnn_stage_3_output)


#         return final_output, qcnn_stage_3_output
    
    
    
    
    
    
class QuantumGenerator(nn.Module):
    def __init__(self, KernelAncilla, layer_num, 
                 stage_1_dim, stage_1_input_size, stage_1_kernel_size, stage_1_stride, stage_1_padding, 
                 stage_2_dim, stage_2_input_size, stage_2_kernel_size, stage_2_stride, stage_2_padding,
                 stage_3_dim, stage_3_input_size, stage_3_kernel_size, stage_3_stride, stage_3_padding,
                 stage_4_dim, stage_4_input_size, stage_4_kernel_size, stage_4_stride, stage_4_padding,
                 final_dim, final_size):  
        
        super().__init__()

        self.KernelAncilla = KernelAncilla


        self.BatchNorm2d_stage_2_dim = nn.BatchNorm2d(stage_2_dim)
        self.BatchNorm2d_stage_3_dim = nn.BatchNorm2d(stage_3_dim)
        self.BatchNorm2d_stage_4_dim = nn.BatchNorm2d(stage_4_dim)
        self.BatchNorm2d_final_dim = nn.BatchNorm2d(final_dim)


        self.ReLU = nn.ReLU(True)
        self.Tanh = nn.Tanh()
        
        
        self.stage_1_dim = stage_1_dim
        self.stage_1_input_size = stage_1_input_size
        self.stage_1_kernel_size = stage_1_kernel_size
        self.stage_1_stride = stage_1_stride
        self.stage_1_padding = stage_1_padding
        
        self.stage_2_dim = stage_2_dim
        self.stage_2_input_size = stage_2_input_size
        self.stage_2_kernel_size = stage_2_kernel_size
        self.stage_2_stride = stage_2_stride
        self.stage_2_padding = stage_2_padding
                 
        self.stage_3_dim = stage_3_dim
        self.stage_3_input_size = stage_3_input_size
        self.stage_3_kernel_size = stage_3_kernel_size
        self.stage_3_stride = stage_3_stride
        self.stage_3_padding = stage_3_padding   

        self.stage_4_dim = stage_4_dim
        self.stage_4_input_size = stage_4_input_size
        self.stage_4_kernel_size = stage_4_kernel_size
        self.stage_4_stride = stage_4_stride
        self.stage_4_padding = stage_4_padding    
        
            
        self.final_dim = final_dim
        self.final_size = final_size

        
        self.kernel_qubit_1 = KernelAncilla + int(np.ceil(math.log2(stage_1_kernel_size[0]*stage_1_kernel_size[1]))) 
        self.kernel_qubit_2 = KernelAncilla + int(np.ceil(math.log2(stage_2_kernel_size[0]*stage_2_kernel_size[1]))) 
        self.kernel_qubit_3 = KernelAncilla + int(np.ceil(math.log2(stage_3_kernel_size[0]*stage_3_kernel_size[1]))) 
        self.kernel_qubit_4 = KernelAncilla + int(np.ceil(math.log2(stage_4_kernel_size[0]*stage_4_kernel_size[1]))) 
        
        # -------------------------------- stage 1: 10*1*1 --> 16*4*4 --------------------------------
        
        self.stage_1_qcnn_node = create_qcnn_node(KernelAncilla=KernelAncilla,
                                                  output_size=stage_2_input_size, 
                                                  kernel_size=stage_1_kernel_size,
                                                  layer_num=layer_num)
        self.tmp_layer_1 = []
        for ii in range(stage_1_dim):
            self.tmp_layer_1.append(qml.qnn.TorchLayer(self.stage_1_qcnn_node, {"weights": (layer_num, self.kernel_qubit_1)}))

        self.qgen_stage_1 = nn.Sequential(*self.tmp_layer_1)
        
        
        
        
        # -------------------------------- stage 2: 16*4*4 --> 8*10*10 --------------------------------

        self.stage_2_qcnn_node = create_qcnn_node(KernelAncilla=KernelAncilla,
                                                  output_size=stage_3_input_size, 
                                                  kernel_size=stage_2_kernel_size, 
                                                  layer_num=layer_num)
        self.tmp_layer_2 = []
        for ii in range(stage_2_dim):
            self.tmp_layer_2.append(qml.qnn.TorchLayer(self.stage_2_qcnn_node, {"weights": (layer_num, self.kernel_qubit_2)}))
                
        self.qgen_stage_2 = nn.Sequential(*self.tmp_layer_2)



        # -------------------------------- stage 3: 8*10*10 --> 4*13*13 --------------------------------

        self.stage_3_qcnn_node = create_qcnn_node(KernelAncilla=KernelAncilla,
                                                  output_size=stage_4_input_size, 
                                                  kernel_size=stage_3_kernel_size, 
                                                  layer_num=layer_num)
        self.tmp_layer_3 = []
        for ii in range(stage_3_dim):
            self.tmp_layer_3.append(qml.qnn.TorchLayer(self.stage_3_qcnn_node, {"weights": (layer_num, self.kernel_qubit_3)}))
                
        self.qgen_stage_3 = nn.Sequential(*self.tmp_layer_3)
        
        
        
        # -------------------------------- stage 4: 4*13*13 --> 1*16*16 --------------------------------

        self.stage_4_qcnn_node = create_qcnn_node(KernelAncilla=KernelAncilla,
                                                  output_size=final_size, 
                                                  kernel_size=stage_4_kernel_size,
                                                  layer_num=layer_num)
        self.tmp_layer_4 = []
        for ii in range(stage_4_dim):
            self.tmp_layer_4.append(qml.qnn.TorchLayer(self.stage_4_qcnn_node, {"weights": (layer_num, self.kernel_qubit_4)}))
                
        self.qgen_stage_4 = nn.Sequential(*self.tmp_layer_4)

    
    
    def forward(self, z):
        
        # do qcnn and activation function on all the data in this batch
        # z: (batch_size, latent_dim, 1, 1)

        # -------------------------------- stage 1: 10*1*1 --> 16*4*4 --------------------------------

        # qcnn_stage_1_output: (batch_size, stage_2_dim, stage_2_input_size[0], stage_2_input_size[1])
        qcnn_stage_1_output = qcnn(KernelAncilla=self.KernelAncilla, input_feature=z, 
                                   output_dim=self.stage_2_dim, output_size=self.stage_2_input_size, 
                                   kernel_size=self.stage_1_kernel_size, stride=self.stage_1_stride, 
                                   padding_index=self.stage_1_padding, qcnn_layer=self.qgen_stage_1)
        
        # qcnn_stage_2_input: (batch_size, stage_2_dim, stage_2_input_size[0], stage_2_input_size[1])
        qcnn_stage_2_input = self.ReLU(self.BatchNorm2d_stage_2_dim(qcnn_stage_1_output))
        
        # -------------------------------- stage 2: 16*4*4 --> 8*10*10 --------------------------------

        # qcnn_stage_2_output: (batch_size, stage_3_dim, stage_3_input_size[0], stage_3_input_size[1])
        qcnn_stage_2_output = qcnn(KernelAncilla=self.KernelAncilla, input_feature=qcnn_stage_2_input, 
                                   output_dim=self.stage_3_dim, output_size=self.stage_3_input_size, 
                                   kernel_size=self.stage_2_kernel_size, stride=self.stage_2_stride, 
                                   padding_index=self.stage_2_padding, qcnn_layer=self.qgen_stage_2)
        

        # qcnn_stage_3_input: (batch_size, stage_3_dim, stage_3_input_size[0], stage_3_input_size[1])
        qcnn_stage_3_input = self.ReLU(self.BatchNorm2d_stage_3_dim(qcnn_stage_2_output))



        # -------------------------------- stage 3: 8*10*10 --> 4*13*13 --------------------------------

        # qcnn_stage_3_output: (batch_size, stage_4_dim, stage_4_input_size[0], stage_4_input_size[1])
        qcnn_stage_3_output = qcnn(KernelAncilla=self.KernelAncilla, input_feature=qcnn_stage_3_input, 
                                   output_dim=self.stage_4_dim, output_size=self.stage_4_input_size, 
                                   kernel_size=self.stage_3_kernel_size, stride=self.stage_3_stride, 
                                   padding_index=self.stage_3_padding, qcnn_layer=self.qgen_stage_3)

        # qcnn_stage_4_input: (batch_size, stage_4_dim, stage_4_input_size[0], stage_4_input_size[1])
        qcnn_stage_4_input = self.ReLU(self.BatchNorm2d_stage_4_dim(qcnn_stage_3_output))
        
        # -------------------------------- stage 4: 4*13*13 --> 1*16*16 --------------------------------

        # qcnn_stage_4_output: (batch_size, final_dim, final_size[0], final_size[1])
        qcnn_stage_4_output = qcnn(KernelAncilla=self.KernelAncilla, input_feature=qcnn_stage_4_input, 
                                   output_dim=self.final_dim, output_size=self.final_size, 
                                   kernel_size=self.stage_4_kernel_size, stride=self.stage_4_stride, 
                                   padding_index=self.stage_4_padding, qcnn_layer=self.qgen_stage_4)
                
        qcnn_stage_4_output_add_lastBN = self.BatchNorm2d_final_dim(qcnn_stage_4_output)


        # final_output: (batch_size, final_dim, final_size[0], final_size[1])
        final_output = self.Tanh(qcnn_stage_4_output_add_lastBN)


        return final_output, qcnn_stage_4_output, qcnn_stage_4_output_add_lastBN
    
    
        
        

    
    