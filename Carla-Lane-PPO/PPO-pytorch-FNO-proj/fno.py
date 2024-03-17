import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# import matplotlib.pyplot as plt 
import numpy as np


class SpectralConv2d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d_fast, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """
        # input shape : (4, 64, 84, 84)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        
        # print(self.weights1.shape)
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # print(batchsize)
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)
        # print(x_ft.shape)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        # print(out_ft.shape)   #torch.Size([4, 128, 84, 43])
        # print("Hey",self.weights1.shape)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width=128, num_actions=5):#history
        super(FNO2d, self).__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 2 # pad the domain if input is non-periodic

        self.fc0 = nn.Linear(6, width // 2) # NOTE : Changes for channel pool
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
    
        self.conv0 = SpectralConv2d_fast(width // 2, width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d_fast(width, width, self.modes1, self.modes2) # 64, 128
        self.conv2 = SpectralConv2d_fast(width, width * 4, self.modes1, self.modes2) # 128, 64
        
        # NOTE : Changes for channel pool
        self.w0 = nn.Conv2d(width // 2, width, 1)
        self.w1 = nn.Conv2d(width, width, 1)
        self.w2 = nn.Conv2d(width, width * 4, 1)
   
        # self.fc1 = nn.Linear(width * 2, width)
        # self.fc1 = nn.Linear(1600, 512)
        # self.fc2 = nn.Linear(width, num_actions)

    def forward(self, x):
        # print(x.shape)
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        if x.shape[1] == 4:
            x = x.permute(0, 2, 3, 1)
        # print(x.shape)
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)

        x = x.permute(0, 3, 1, 2)
        # x = F.pad(x, [0,self.padding, 0,self.padding]) # pad the domain if input is non-periodic
        # print(x.shape)
        
        x1 = self.conv0(x)
        x2 = self.w0(x)

        
        x = x1 + x2
        # print(x.shape)
 
        x = F.gelu(self.pool(x))
      
        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(self.pool(x))
    
        
        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(self.pool(x))

        # print(x.shape)
        x = x.max(dim=-1)[0].max(dim=-1)[0]
        # print(x.shape)
        
        # x = x.view(x.shape[0], -1)
        
        # # import sys; sys.exit()
        # # x = x[..., :-self.padding, :-self.padding] # pad the domain if input is non-periodic
        # x = self.fc1(x)
        # x = F.gelu(x)
        # x = torch.log_softmax(self.fc2(x), dim=-1)
        # import sys; sys.exit()
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)


if __name__ == "__main__":
    model = FNO2d(5, 5, 128)
    # total_params = sum(
	#     param.numel() for param in model.parameters()
    # )
    # for param in model.parameters():
    #     print(param.numel())
    # print(total_params) #38_06_757, 15_53_893

    inp = torch.randn(4, 4, 84, 84)
    inp = inp.permute(0, 2, 3, 1)   
    print(inp.shape)   # batch , height width , channel
    output = model(inp)
    print(output.shape)

