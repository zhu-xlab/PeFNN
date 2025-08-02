import torch.nn.functional as F
import torch
import torch.nn as nn
import math
from utils import grid

class MCFL(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True,
                 spectral=False, Conservation=False, Multiple_Rotation=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rt_group_size = 4
        self.group_size = self.rt_group_size
        assert kernel_size % 2 == 1, "kernel size must be odd"
        dtype = torch.cfloat if spectral else torch.float
        self.kernel_size_Y = kernel_size
        self.kernel_size_X = kernel_size // 2 + 1 if Conservation else kernel_size
        self.Conservation = Conservation
        self.Multiple_Rotation = Multiple_Rotation
        if self.Conservation: # single rotation kernel
            if self.Multiple_Rotation:
                self.W = nn.ParameterDict({
                    'y0_modes': torch.nn.Parameter(torch.empty(out_channels, 1, in_channels, self.group_size, self.kernel_size_X - 1, self.kernel_size_X - 1, dtype=dtype)),
                    'yposx_modes': torch.nn.Parameter(torch.empty(out_channels, 1, in_channels, self.group_size, self.kernel_size_Y, 1, dtype=dtype)),
                    '00_modes': torch.nn.Parameter(torch.empty(out_channels, 1, in_channels, self.group_size, 1, self.kernel_size_X - 1, dtype=torch.float))
                })
            else:
                self.W = nn.ParameterDict({
                'y0_modes': torch.nn.Parameter(torch.empty(out_channels, 1, in_channels, self.group_size, self.kernel_size_X - 1, 1, dtype=dtype)),
                'yposx_modes': torch.nn.Parameter(torch.empty(out_channels, 1, in_channels, self.group_size, self.kernel_size_Y, self.kernel_size_X - 1, dtype=dtype)),
                '00_modes': torch.nn.Parameter(torch.empty(out_channels, 1, in_channels, self.group_size, 1, 1, dtype=torch.float))
                })
        else:
            self.W = nn.Parameter(torch.empty(out_channels, 1, in_channels, self.group_size, self.kernel_size_Y, self.kernel_size_X, dtype=dtype))
        self.B = nn.Parameter(torch.empty(1, out_channels, 1, 1)) if bias else None
        self.eval_build = True
        self.reset_parameters()
        self.get_weight()

    def reset_parameters(self):
        if self.Conservation:
            for v in self.W.values():
                nn.init.kaiming_uniform_(v, a=math.sqrt(5))
        else:
            nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        if self.B is not None:
            nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))

    def get_weight(self):

        if self.training:
            self.eval_build = True
        elif self.eval_build:
            self.eval_build = False
        else:
            return

        if self.Conservation:
            if self.Multiple_Rotation:
                self.weights1 = torch.cat([self.W["y0_modes"], self.W["00_modes"].cfloat(), self.W["y0_modes"].rot90(k=1, dims=[-2, -1])], dim=-2)
                self.weights2 = torch.cat([self.W["y0_modes"].conj().rot90(k=3, dims=[-2, -1]), self.W["00_modes"].cfloat(),
                                           self.W["y0_modes"].conj().rot90(k=2, dims=[-2, -1])], dim=-2)
                self.weights = torch.cat([self.weights1, self.W["yposx_modes"].cfloat(), self.weights2], dim=-1)
            else:
                self.weights = torch.cat([self.W["y0_modes"], self.W["00_modes"].cfloat(), self.W["y0_modes"].conj()], dim=-2)
                self.weights = torch.cat([self.weights, self.W["yposx_modes"]], dim=-1)
                self.weights = torch.cat([self.weights[..., 1:].conj().rot90(k=2, dims=[-2, -1]), self.weights], dim=-1)
        else:
            self.weights = self.W[:]

        self.weights = self.weights.repeat(1, self.group_size, 1, 1, 1, 1)

        # apply elements in the rotation group
        for k in range(1, self.rt_group_size):
            self.weights[:, k] = self.weights[:, k - 1].rot90(dims=[-2, -1])
            self.weights[:, k] = torch.cat([self.weights[:, k, :, -1].unsqueeze(2), self.weights[:, k, :, :-1]], dim=2)

        self.weights = self.weights.view(self.out_channels * self.group_size, self.in_channels * self.group_size,
                                         self.kernel_size_Y, self.kernel_size_Y)
        if self.B is not None:
            self.bias = self.B.repeat_interleave(repeats=self.group_size, dim=1)

        if self.Conservation:
            self.weights = self.weights[..., -self.kernel_size_X:]

    def forward(self, x):

        self.get_weight()

        # output is of shape (batch * out_channels, number of group elements, ny, nx)
        x = nn.functional.conv2d(input=x, weight=self.weights)

        # add the bias
        if self.B is not None:
            x = x + self.bias
        return x

class GSpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super(GSpectralConv2d, self).__init__()

        """
        2D Fourier layer
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.conv = MCFL(in_channels=in_channels, out_channels=out_channels, kernel_size=2 * modes - 1, bias=False, spectral=True, Conservation=True)
        self.get_weight()

    # Building the weight
    def get_weight(self):
        self.conv.get_weight()
        self.weights = self.conv.weights.transpose(0, 1)

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]

        # get the index of the zero frequency and construct weight
        freq0_y = (torch.fft.fftshift(torch.fft.fftfreq(x.shape[-2])) == 0).nonzero().item()
        self.get_weight()

        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.fftshift(torch.fft.rfft2(x), dim=-2)
        x_ft = x_ft[..., (freq0_y - self.modes + 1):(freq0_y + self.modes), :self.modes]

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.weights.shape[0], x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat,
                             device=x.device)
        out_ft[..., (freq0_y - self.modes + 1):(freq0_y + self.modes), :self.modes] = \
            self.compl_mul2d(x_ft, self.weights)

        # Return to physical space
        x = torch.fft.irfft2(torch.fft.ifftshift(out_ft, dim=-2), s=(x.size(-2), x.size(-1)))

        return x

class GMLP2d(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(GMLP2d, self).__init__()
        self.mlp1 = MCFL(in_channels=in_channels, out_channels=mid_channels, kernel_size=1)
        self.mlp2 = MCFL(in_channels=mid_channels, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x

class MLP2d(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP2d, self).__init__()
        self.mlp1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv2d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x

class GNorm(nn.Module):
    def __init__(self, width, group_size):
        super().__init__()
        self.group_size = group_size
        self.norm = torch.nn.InstanceNorm3d(width)

    def forward(self, x):
        x = x.view(x.shape[0], -1, self.group_size, x.shape[-2], x.shape[-1])
        x = self.norm(x)
        x = x.view(x.shape[0], -1, x.shape[-2], x.shape[-1])
        return x

class PeFNN(nn.Module):
    def __init__(self, num_channels, modes, width, initial_step, grid_type):
        super(PeFNN, self).__init__()

        """
        The overall network. It contains 4 layers of the MC-Fourier layer.
        """

        self.dt = 30 # dt = 30 for flood simulation, dt = 0.01 for SWE and NS
        self.modes = modes
        self.width = width

        self.grid = grid(twoD=True, grid_type=grid_type)
        self.p = nn.Linear(num_channels * initial_step + self.grid.grid_dim, self.width * 4)
        self.conv0 = GSpectralConv2d(in_channels=self.width, out_channels=self.width, modes=self.modes)
        self.conv1 = GSpectralConv2d(in_channels=self.width, out_channels=self.width, modes=self.modes)
        self.conv2 = GSpectralConv2d(in_channels=self.width, out_channels=self.width, modes=self.modes)
        self.conv3 = GSpectralConv2d(in_channels=self.width, out_channels=self.width, modes=self.modes)
        self.mlp0 = GMLP2d(in_channels=self.width, out_channels=self.width, mid_channels=self.width)
        self.mlp1 = GMLP2d(in_channels=self.width, out_channels=self.width, mid_channels=self.width)
        self.mlp2 = GMLP2d(in_channels=self.width, out_channels=self.width, mid_channels=self.width)
        self.mlp3 = GMLP2d(in_channels=self.width, out_channels=self.width, mid_channels=self.width)
        self.w0 = MCFL(in_channels=self.width, out_channels=self.width, kernel_size=1)
        self.w1 = MCFL(in_channels=self.width, out_channels=self.width, kernel_size=1)
        self.w2 = MCFL(in_channels=self.width, out_channels=self.width, kernel_size=1)
        self.w3 = MCFL(in_channels=self.width, out_channels=self.width, kernel_size=1)
        self.norm = GNorm(self.width, group_size=4)
        self.q = MLP2d(self.width * 4, num_channels, self.width)  # output channel is 1: u(x, y)

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1], x.shape[2], -1)
        h = x.permute(0, 3, 1, 2)
        u_prev = h[:, 0:1, ...]
        x = self.grid(x)
        x = self.p(x)
        x = x.permute(0, 3, 1, 2)

        x1 = self.norm(self.conv0(self.norm(x)))
        x1 = self.mlp0(x1)
        x2 = self.w0(x)
        x = x1 + x2
        xm1 = x

        x1 = self.norm(self.conv1(self.norm(x)))
        x1 = self.mlp1(x1)
        x2 = self.w1(x)
        x = x1 + x2
        xm2 = x

        x1 = self.norm(self.conv2(self.norm(x)))
        x1 = self.mlp2(x1)
        x2 = self.w2(x)
        x = x1 + x2
        xm3 = x

        x1 = self.norm(self.conv3(self.norm(x)))
        x1 = self.mlp3(x1)
        x2 = self.w3(x)
        x = x1 + x2
        xm4 = x

        # x = x[..., :-self.padding, :-self.padding] # pad the domain if input is non-periodic
        u_res = self.q(xm1 * xm2 * xm3 * xm4)
        u_next = u_prev + self.dt * u_res
        u_next = u_next.permute(0, 2, 3, 1)
        return u_next.unsqueeze(-2)