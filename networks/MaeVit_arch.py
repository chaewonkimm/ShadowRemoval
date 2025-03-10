import os, sys, time
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
cwd = os.getcwd()
print("Current working directory: {0}".format(cwd))

from functools import partial
import torch
import torch.nn as nn
# from timm.models.vision_transformer import Block
from networks.vit import Block
from networks.Patch_embed import PatchEmbed
import torch.nn.functional as F


class PyramidPooling(nn.Module):
    def __init__(self, in_channels, out_channels, num_scales=4, ct_channels=1):
        super().__init__()
        if num_scales == 4:
            scales = (4, 8, 16, 32)
        elif num_scales == 3:
            scales = (4, 8, 16)
        self.stages = nn.ModuleList([self._make_stage(in_channels, scale, ct_channels) for scale in scales])
        self.bottleneck = nn.Conv2d(in_channels + len(scales) * ct_channels, out_channels, kernel_size=1, stride=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def _make_stage(self, in_channels, scale, ct_channels):
        prior = nn.AvgPool2d(kernel_size=(scale, scale))
        conv = nn.Conv2d(in_channels, ct_channels, kernel_size=1, bias=False)
        relu = nn.LeakyReLU(0.2, inplace=True)
        return nn.Sequential(prior, conv, relu)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = torch.cat(
            [F.interpolate(stage(feats), size=(h, w), mode='nearest') for stage in self.stages] + [feats],
            dim=1
        )
        return self.relu(self.bottleneck(priors))


class MaskedAutoencoderViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, out_chans=3, fea_chans=16, num_scales=4,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, global_residual=False):
        super().__init__()
        self.global_residual = global_residual

        self.patch_embed = PatchEmbed(patch_size, in_chans, embed_dim)

        self.freq_fc = nn.Linear(in_chans, embed_dim)
        self.film_gamma = nn.Linear(embed_dim, embed_dim)
        self.film_beta = nn.Linear(embed_dim, embed_dim)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.decoder_embed_for_unselected = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * fea_chans, bias=True)
        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

        self.pyramid_module = PyramidPooling(fea_chans, fea_chans, num_scales=num_scales, ct_channels=fea_chans // 4)
        self.last_conv = nn.Conv2d(fea_chans, out_chans, kernel_size=3, padding=1, stride=1, bias=False)

    def initialize_weights(self):
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def unpatchify(self, x, H, W):
        p = self.patch_embed.patch_size[0]
        h = H // p
        w = W // p
        assert h * w == x.shape[1]
        x = x.reshape(x.shape[0], h, w, p, p, -1)
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(x.shape[0], -1, h * p, w * p)
        return imgs

    def forward_encoder(self, imgs):
        x = self.patch_embed(imgs)
        fft_result = torch.fft.fft2(imgs, dim=(-2, -1))
        freq_magnitude = torch.abs(fft_result)
        freq_vector = freq_magnitude.mean(dim=(-2, -1))
        freq_repr = self.freq_fc(freq_vector)
        gamma = self.film_gamma(freq_repr).unsqueeze(1)
        beta = self.film_beta(freq_repr).unsqueeze(1)
        x = gamma * x + beta
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

    def forward_decoder(self, x):
        x = self.decoder_embed(x)
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        return x

    def forward(self, imgs):
        _, _, ori_H, ori_W = imgs.size()
        latent = self.forward_encoder(imgs)
        pred = self.forward_decoder(latent)
        pred_wOri_Size = self.unpatchify(pred, ori_H, ori_W)
        pred_wOri_Size = self.last_conv(self.pyramid_module(pred_wOri_Size))
        if self.global_residual:
            pred_wOri_Size = pred_wOri_Size + imgs
        return pred_wOri_Size

def mae_vit_small_patch16_dec128d4b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=8, embed_dim=256, depth=6, num_heads=8,
        decoder_embed_dim=128, decoder_depth=1, decoder_num_heads=4,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    return model

class PyramidPooling(nn.Module):
    def __init__(self, in_channels, out_channels, num_scales=4, ct_channels=1):
        super().__init__()
        if num_scales == 4:
            scales = (4, 8, 16, 32)
        elif num_scales == 3:
            scales = (4, 8, 16)
        self.stages = nn.ModuleList([self._make_stage(in_channels, scale, ct_channels) for scale in scales])
        self.bottleneck = nn.Conv2d(in_channels + len(scales) * ct_channels, out_channels, kernel_size=1, stride=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
    def _make_stage(self, in_channels, scale, ct_channels):
        prior = nn.AvgPool2d(kernel_size=(scale, scale))
        conv = nn.Conv2d(in_channels, ct_channels, kernel_size=1, bias=False)
        relu = nn.LeakyReLU(0.2, inplace=True)
        return nn.Sequential(prior, conv, relu)
    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = torch.cat(
            [F.interpolate(stage(feats), size=(h, w), mode='nearest') for stage in self.stages] + [feats],
            dim=1
        )
        return self.relu(self.bottleneck(priors))

if __name__ == "__main__":
    model = mae_vit_small_patch16_dec128d4b(img_size=256)
    print('#generator parameters:', sum(p.numel() for p in model.parameters()))
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print(y.shape)
