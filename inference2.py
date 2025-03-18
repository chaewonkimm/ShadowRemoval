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
from networks.Split_images import process_split_image_with_shadow_matte
from PIL import Image
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def split_image_overlap(img, crop_size, overlap_size):
    B, C, H, W = img.shape
    stride = crop_size - overlap_size
    y_starts = list(range(0, H - crop_size + 1, stride))
    if y_starts and y_starts[-1] != H - crop_size:
        y_starts.append(H - crop_size)
    x_starts = list(range(0, W - crop_size + 1, stride))
    if x_starts and x_starts[-1] != W - crop_size:
        x_starts.append(W - crop_size)
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

def sliding_crop_left(img, patch_size, stride):
    B, C, H, W = img.shape
    patches = []
    positions = []
    for x in range(0, W - patch_size + 1, stride):
        patch = img[:, :, 0:H, x:x+patch_size]
        patches.append(patch)
        positions.append((0, x, patch_size, H))
    return patches, positions

def merge_sliding_crops(crops, crop_positions, original_width, overlap_size):
    B, C, H, W_crop = crops[0].shape
    device = crops[0].device
    merged = torch.zeros((B, C, H, original_width), device=device)
    weight = torch.zeros((B, 1, H, original_width), device=device)
    for crop, pos in zip(crops, crop_positions):
        x = pos[1]
        merged[:, :, :, x:x+W_crop] += crop
        weight[:, :, :, x:x+W_crop] += 1.0
    merged = merged / (weight + 1e-8)
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
    parser = argparse.ArgumentParser(description="Inference: ViT+NAFNet pipeline with matte predictor and sliding crop")
    parser.add_argument("--checkpoint", type=str, default="../autodl-tmp/SR_1/train_nafnet_wmatte_sliding_matte/train_nafnet_wmatte.pth")
    parser.add_argument("--vit_checkpoint", type=str, default="./pretrained/vit_matte_weight0.1_finetuning.pth")
    parser.add_argument("--input_dir", type=str, default="../data/test/")
    parser.add_argument("--output_dir", type=str, default="./outputs_overlap")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--nafn_patch_size", type=int, default=256)
    parser.add_argument("--overlap_size", type=int, default=128)
    parser.add_argument("--crop_stride", type=int, default=250)
    parser.add_argument("--img_size", type=int, default=1000)
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
    
    s_time = time.time()
    with torch.no_grad():
        for data, fname in loader:
            inputs = data.to(device) 
            B, C, H, W = inputs.shape
            crops, crop_positions = sliding_crop_left(inputs, patch_size=args.img_size, stride=args.crop_stride)
            processed_crops = []

            for crop in crops:
                sub_data, positions = split_image_overlap(crop, crop_size=args.nafn_patch_size, overlap_size=args.overlap_size)
                matte_patches = []
                for sub in sub_data:
                    matte_patch = matte_predictor.predict_from_tensor(sub)
                    matte_patches.append(matte_patch)
                processed = process_split_image_with_shadow_matte(sub_data, matte_patches, net)
                crop_processed = merge_image_overlap(processed, positions, crop_size=args.nafn_patch_size,
                                                     resolution=crop.shape, overlap_size=args.overlap_size, blend_mode='gaussian')
                processed_crops.append(crop_processed)
            outputs = merge_sliding_crops(processed_crops, crop_positions, original_width=W, overlap_size=args.overlap_size)
            nafnet_output = net_1(outputs)
            nafnet_output = torch.clamp(nafnet_output, 0, 1)
            for i in range(nafnet_output.size(0)):
                out_path = os.path.join(args.output_dir, fname[i])
                torchvision.utils.save_image(nafnet_output[i].cpu(), out_path)
                print(f"Saved output image: {out_path}")
    e_time = time.time()
    print(f"Elapsed time: {e_time - s_time} seconds")

if __name__ == "__main__":
    from networks.shadow_matte import ShadowMattePredictor
    matte_predictor = ShadowMattePredictor(
        model_path="pretrained/epoch_1400.pth",
        img_size=256,
        device=device
    )
    main()
