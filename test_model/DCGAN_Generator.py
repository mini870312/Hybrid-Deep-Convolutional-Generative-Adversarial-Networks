import torch
import torch.nn as nn


latent_dim = 10
mid_dim_1 = 4
mid_dim_2 = 8
mid_dim_3 = 16



# Generator Code

class Generator_4_1(nn.Module):
    def __init__(self):
        super().__init__()

        self.main = nn.Sequential(
            nn.ConvTranspose2d( latent_dim, mid_dim_3, 4, 1, 0, bias=False),
            nn.BatchNorm2d(mid_dim_3),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d( mid_dim_3, mid_dim_2, 7, 1, 0, bias=False),
            nn.BatchNorm2d(mid_dim_2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(  mid_dim_2, mid_dim_1, 4, 1, 0, bias=False),
            nn.BatchNorm2d(mid_dim_1),
            nn.ConvTranspose2d(  mid_dim_1, 1, 4, 1, 0, bias=False),
            nn.ReLU(inplace=True),
            nn.Tanh()
        )

    def forward(self, input):
        
        return self.main(input)




# Generator Code

class Generator_7_1(nn.Module):
    def __init__(self):
        super().__init__()

        self.main = nn.Sequential(
            nn.ConvTranspose2d( latent_dim, mid_dim_3, 7, 1, 0, bias=False),
            nn.BatchNorm2d(mid_dim_3),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d( mid_dim_3, mid_dim_2, 4, 1, 0, bias=False),
            nn.BatchNorm2d(mid_dim_2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(  mid_dim_2, mid_dim_1, 4, 1, 0, bias=False),
            nn.BatchNorm2d(mid_dim_1),
            nn.ConvTranspose2d(  mid_dim_1, 1, 4, 1, 0, bias=False),
            nn.ReLU(inplace=True),
            nn.Tanh()
        )

    def forward(self, input):
        
        return self.main(input)



    