"""Network factory for creating different UNet architectures."""

from .unet import UNet as unet
from .unet_proj import UNetProj as unet_proj

def net_factory(net_type, in_channels, num_classes, **kwargs):
    return globals()[net_type](in_channels=in_channels, num_classes=num_classes, **kwargs).cuda()
