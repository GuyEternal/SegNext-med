#%%
import yaml, math, os
with open('config.yaml') as fh:
    config = yaml.load(fh, Loader=yaml.FullLoader)
import torch
import torch.nn.functional as F
import torch.nn as nn

from backbone import MSCANet
# from decoder import HamDecoder
from cross_attention import CrossDecoder


class CrossNet(nn.Module):
    def __init__(self, num_classes, in_channnels=3, embed_dims=[64, 128, 320, 512],
                 ffn_ratios=[8, 8, 4, 4], depths=[3, 3, 12, 3], num_stages=4,
                 dec_outChannels=64, config=config, dropout=0.0, drop_path=0.1):
        super().__init__()
        # Classification head is now integrated into the decoder
        self.encoder = MSCANet(in_channnels=in_channnels, embed_dims=embed_dims,
                           ffn_ratios=ffn_ratios, depths=depths, num_stages=num_stages,
                           drop_path=drop_path)
        self.decoder = CrossDecoder(
            outChannels=dec_outChannels, config=config, enc_embed_dims=embed_dims)
        self.init_weights()

        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, val=1.0)
                nn.init.constant_(m.bias, val=0.0)
            if isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                nn.init.normal_(m.weight, std=math.sqrt(2.0/fan_out), mean=0)

    def forward(self, x):
        enc_feats = self.encoder(x)
        output = self.decoder(enc_feats)  # Decoder now includes classification head
        
        # Resize to match input resolution
        output = F.interpolate(output, size=x.size()[-2:], mode='bilinear', align_corners=True)
        
        return output

