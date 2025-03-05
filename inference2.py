import os
import time
import sys
import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from functools import partial
from torchvision.transforms import ToPILImage
from torchvision import transforms
from networks.MaeVit_arch import MaskedAutoencoderViT
from networks.NAFNet_arch import NAFNet
from PIL import Image
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def split_image_overlap(img, crop_size, overlap_size):
    B, C, H, W = img.shape
    stride = crop_size - overlap_size
    y_starts = list(range(0, H - crop_size + 1, stride))
    if y_starts and y_starts[-1] != H - crop_size:
        y_starts.append(H - crop_size)
    elif not y_starts:
        y_starts = [0]
    x_starts = list(range(0, W - crop_size + 1, stride))
    if x_starts and x_starts[-1] != W - crop_size:
        x_starts.append(W - crop_size)
    elif not x_starts:
        x_starts = [0]
    patches = []
    positions = []
    for y in y_starts:
        for x in x_starts:
            patch = img[:, :, y:y+crop_size, x:x+crop_size]
            patches.append(patch)
            positions.append((y, x, crop_size, crop_size))
    return patches, positions

def create_gaussian_mask(patch_size, overlap):
    h, w = patch_size
    weight_y = torch.ones(h, dtype=torch.float32)
    sigma = overlap / 2.0 if overlap > 0 else 1.0
    for i in range(h):
        if i < overlap:
            weight_y[i] = math.exp(-0.5 * ((overlap - i)/sigma)**2)
        elif i > h - overlap - 1:
            weight_y[i] = math.exp(-0.5 * ((i - (h - overlap - 1))/sigma)**2)
    weight_x = torch.ones(w, dtype=torch.float32)
    for j in range(w):
        if j < overlap:
            weight_x[j] = math.exp(-0.5 * ((overlap - j)/sigma)**2)
        elif j > w - overlap - 1:
            weight_x[j] = math.exp(-0.5 * ((j - (w - overlap - 1))/sigma)**2)
    mask = torch.ger(weight_y, weight_x)
    return mask.unsqueeze(0).unsqueeze(0)

def merge_image_overlap(patches, positions, crop_size, resolution, overlap_size, blend_mode='gaussian'):
    B, C, H, W = resolution
    device = patches[0].device
    merged = torch.zeros((B, C, H, W), device=device)
    weight_sum = torch.zeros((B, 1, H, W), device=device)
    for patch, pos in zip(patches, positions):
        y, x, ph, pw = pos
        if blend_mode == 'gaussian' and overlap_size > 0:
            mask = create_gaussian_mask((ph, pw), overlap_size).to(device)
        else:
            mask = torch.ones((1, 1, ph, pw), device=device)
        merged[:, :, y:y+ph, x:x+pw] += patch * mask
        weight_sum[:, :, y:y+ph, x:x+pw] += mask
    merged = merged / (weight_sum + 1e-8)
    return merged

class InferenceDataset(torch.utils.data.Dataset):
    def __init__(self, dir_path, transform=None):
        self.dir_path = dir_path
        self.image_names = sorted(os.listdir(dir_path))
        self.transform = transform if transform is not None else transforms.Compose([transforms.ToTensor()])
    def __len__(self):
        return len(self.image_names)
    def __getitem__(self, idx):
        img_path = os.path.join(self.dir_path, self.image_names[idx])
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.image_names[idx]

def parse_args():
    parser = argparse.ArgumentParser(description="Inference: ViT+NAFNet with overlap for NAFNet only")
    parser.add_argument("--checkpoint", type=str, default="../autodl-tmp/SR_1/train_shadow_nafnet_32_overlap/nafnet_stage2_wloss_save_overlap.pth")
    parser.add_argument("--vit_checkpoint", type=str, default="./checkpoints/train_vit_stage1_wloss_total/vit_stage1_wloss.pth")
    parser.add_argument("--input_dir", type=str, default="../validation/")
    parser.add_argument("--output_dir", type=str, default="./outputs_overlap2")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--nafn_patch_size", type=int, default=256)
    parser.add_argument("--overlap_size", type=int, default=16)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    net = MaskedAutoencoderViT(
        patch_size=8, embed_dim=256, depth=6, num_heads=8,
        decoder_embed_dim=256, decoder_depth=6, decoder_num_heads=8,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6)
    )
    net_1 = NAFNet(
        img_channel=3, width=24, middle_blk_num=6,
        enc_blk_nums=[1, 1, 1, 28], dec_blk_nums=[1, 1, 1, 1],
        global_residual=False
    )
    net.to(device)
    net_1.to(device)
    net.eval()
    net_1.eval()
    if not os.path.exists(args.vit_checkpoint):
        print(f"[Error] ViT checkpoint not found: {args.vit_checkpoint}")
        sys.exit(1)
    net.load_state_dict(torch.load(args.vit_checkpoint, map_location=device))
    print(f"Loaded ViT checkpoint from {args.vit_checkpoint}")
    if not os.path.exists(args.checkpoint):
        print(f"[Error] NAFNet checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    net_1.load_state_dict(torch.load(args.checkpoint, map_location=device))
    print(f"Loaded NAFNet checkpoint from {args.checkpoint}")
    dataset = InferenceDataset(args.input_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    to_pil = ToPILImage()
    s_time = time.time()
    with torch.no_grad():
        for data, fname in loader:
            inputs = data.to(device)
            orig_H, orig_W = inputs.shape[2], inputs.shape[3]
            patch = 8
            pad_H = (patch - (orig_H % patch)) % patch
            pad_W = (patch - (orig_W % patch)) % patch
            inputs_padded = F.pad(inputs, (0, pad_W, 0, pad_H), mode='reflect')
            vit_output = net(inputs_padded)
            _, _, H_vit, W_vit = vit_output.shape
            sub_images, positions = split_image_overlap(vit_output, crop_size=args.nafn_patch_size, overlap_size=args.overlap_size)
            processed_subs = [net_1(sub) for sub in sub_images]
            merged = merge_image_overlap(processed_subs, positions, crop_size=args.nafn_patch_size,
                                         resolution=(inputs_padded.size(0), inputs_padded.size(1), H_vit, W_vit),
                                         overlap_size=args.overlap_size, blend_mode='gaussian')
            merged = merged[:, :, :orig_H, :orig_W]
            merged = torch.clamp(merged, 0, 1)
            for i in range(merged.size(0)):
                out_path = os.path.join(args.output_dir, fname[i])
                torchvision.utils.save_image(merged[i].cpu(), out_path)
                print(f"Saved output image: {out_path}")
    e_time = time.time()
    print(f"Elapsed time: {e_time - s_time} seconds")

if __name__ == "__main__":
    main()
