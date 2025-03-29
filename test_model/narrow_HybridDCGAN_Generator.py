import torch
import torch.nn as nn
import sys

sys.path.append('/Users/chia-yichou/20231206_macmini_FID_Running/model_preparation/QuantumGAN_utils')
from new_QGAN_twoMid_add_lastBN import QuantumGenerator




device = "cpu"


KernelAncilla = 0
N_layer = 10


# Quantum Generator

def narrow_QuantumGenerator_4_1():

    kernel_size_1 = (7,7)
    kernel_size_2 = (7,7)
    kernel_size_3 = (4,4)

    image_dim = 1
    image_size = (16,16)

    middle_dim_1 = 8
    middle_size_1 = (10,10)

    middle_dim_2 = 16
    middle_size_2 = (4,4)

    final_dim = 1
    final_size = (1,1)

    latent_dim = 10
    latent_size = (1,1)

    # -------------------------------- Generator: 10*1*1 --> 8*4*4 --> 4*10*10 --> 1*16*16 --------------------------------
    narrow_generator_4_1 = QuantumGenerator(KernelAncilla=KernelAncilla, layer_num=N_layer, 
                                stage_1_dim=latent_dim, stage_1_input_size=latent_size, 
                                stage_1_kernel_size=kernel_size_3,stage_1_stride=1, stage_1_padding=3,
                                stage_2_dim=middle_dim_2, stage_2_input_size=middle_size_2, 
                                stage_2_kernel_size=kernel_size_2,stage_2_stride=1, stage_2_padding=6,  
                                stage_3_dim=middle_dim_1, stage_3_input_size=middle_size_1, 
                                stage_3_kernel_size=kernel_size_1,stage_3_stride=1, stage_3_padding=6, 
                                final_dim=image_dim, final_size=image_size).to(device)


    return narrow_generator_4_1




def narrow_QuantumGenerator_7_1():

    kernel_size_1 = (7,7)
    kernel_size_2 = (4,4)
    kernel_size_3 = (7,7)

    image_dim = 1
    image_size = (16,16)

    middle_dim_1 = 8
    middle_size_1 = (10,10)

    middle_dim_2 = 16
    middle_size_2 = (7,7)

    final_dim = 1
    final_size = (1,1)

    latent_dim = 10
    latent_size = (1,1)


    # -------------------------------- Generator: 10*1*1 --> 8*7*7 --> 4*10*10 --> 1*16*16 --------------------------------
    narrow_generator_7_1 = QuantumGenerator(KernelAncilla=KernelAncilla, layer_num=N_layer, 
                                stage_1_dim=latent_dim, stage_1_input_size=latent_size, 
                                stage_1_kernel_size=kernel_size_3,stage_1_stride=1, stage_1_padding=6,
                                stage_2_dim=middle_dim_2, stage_2_input_size=middle_size_2, 
                                stage_2_kernel_size=kernel_size_2,stage_2_stride=1, stage_2_padding=3,  
                                stage_3_dim=middle_dim_1, stage_3_input_size=middle_size_1, 
                                stage_3_kernel_size=kernel_size_1,stage_3_stride=1, stage_3_padding=6, 
                                final_dim=image_dim, final_size=image_size).to(device)

    return narrow_generator_7_1
