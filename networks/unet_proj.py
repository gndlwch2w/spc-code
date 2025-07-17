from torch import nn
from networks.unet import Encoder, Decoder

class UNetProj(nn.Module):
    """UNet with projection head for MoCo-like self-supervised learning."""
    
    def __init__(self, in_channels, num_classes):
        super(UNetProj, self).__init__()

        params = {
            'in_chns': in_channels,
            'feature_chns': [16, 32, 64, 128, 256],
            'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
            'class_num': num_classes,
            'acti_func': 'relu'
        }

        self.encoder = Encoder(params)
        self.decoder = Decoder(params)

        dim = params['feature_chns'][0]
        self.proj_head = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim, affine=False)
        )

    def forward(self, x):
        feature = self.encoder(x)
        output, feat = self.decoder(feature)
        return output, feat
