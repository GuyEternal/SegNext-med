```python
# Update in model.py

# Change this import:
# from decoder import HamDecoder
from crossnext_decoder import CrossNeXtDecoder  # Import CrossNeXtDecoder instead of HamDecoder

# The SegNext class initialization remains the same, but the decoder initialization changes to:
class SegNext(nn.Module):
    def __init__(self, num_classes, in_channnels=3, embed_dims=[32, 64, 460, 256],
                 ffn_ratios=[4, 4, 4, 4], depths=[3, 3, 5, 2], num_stages=4,
                 dec_outChannels=256, config=config, dropout=0.0, drop_path=0.0):
        super().__init__()
        self.cls_conv = nn.Sequential(nn.Dropout2d(p=0.1),
                                      nn.Conv2d(dec_outChannels, num_classes, kernel_size=1))
        self.encoder = MSCANet(in_channnels=in_channnels, embed_dims=embed_dims,
                           ffn_ratios=ffn_ratios, depths=depths, num_stages=num_stages,
                           drop_path=drop_path)
        # Replace HamDecoder with CrossNeXtDecoder
        self.decoder = CrossNeXtDecoder(
            outChannels=dec_outChannels, config=config, enc_embed_dims=embed_dims)
        self.init_weights()

    # The rest of the class definition remains unchanged
    # The forward method stays the same:
    def forward(self, x):
        enc_feats = self.encoder(x)
        dec_out = self.decoder(enc_feats)
        output = self.cls_conv(dec_out)
        output = F.interpolate(output, size=x.size()[-2:], mode='bilinear', align_corners=True)
        return output