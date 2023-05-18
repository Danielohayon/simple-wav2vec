
import torch 
from torch import nn 
import torch.nn.functional as F 
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dropout=0) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride)
        self.dropout = nn.Dropout(p=dropout)
        self.group_norm = nn.GroupNorm()
        self.activation = nn.ReLU()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.dropout(x)
        x = self.group_norm(x)
        x = self.activation(x)
        return x

class Wav2Vec(nn.Module):
    def __init__(self, hidden_size=512, ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        # From paper: The encoder layers have kernel sizes (10, 8, 4, 4, 4) and strides (5, 4, 2, 2, 2).
        # From paper: The layers in both the encoder and context networks consist of a causal convolution with 512 channels
        self.encoder_conv1 = ConvBlock(1,                self.hidden_size, kernel_size=10, stride=5)
        self.encoder_conv2 = ConvBlock(self.hidden_size, self.hidden_size, kernel_size=8, stride=4)
        self.encoder_conv3 = ConvBlock(self.hidden_size, self.hidden_size, kernel_size=4, stride=2)
        self.encoder_conv4 = ConvBlock(self.hidden_size, self.hidden_size, kernel_size=4, stride=2)
        self.encoder_conv5 = ConvBlock(self.hidden_size, self.hidden_size, kernel_size=4, stride=2)
        # From paper: The output of the encoder is a low frequency feature representation zi ∈ Z which
        # encodes about 30 ms of 16 kHz of audio and the striding results in representations zi every 10ms.
        
        # Next, we apply the context network g : Z → C to the output of the encoder network to mix multiple 
        # latent representations zi . . . zi−v into a single contextualized tensor ci = g(zi . . . zi−v ) 
        # for a receptive field size v. The context network has nine layers with kernel size three and stride one. 
        # The total receptive field of the context network is about 210 ms.
        self.context_conv = nn.Sequential(*[ConvBlock(self.hidden_size, self.hidden_size, kernel_size=3, stride=1) for i in range(9)])

    def forward(self, x):
        Z = self.encoder_conv1(x) 
        Z = self.encoder_conv2(Z) 
        Z = self.encoder_conv3(Z) 
        Z = self.encoder_conv4(Z) 
        Z = self.encoder_conv5(Z) 

        C = self.context_conv(Z)
        return Z, C
    
