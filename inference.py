# inference.py
import os
import time
import sys
import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from functools import partial
from torchvision.transforms import ToPILImage
from torchvision import transforms
from networks.MaeVit_arch import MaskedAutoencoderViT
from networks.Split_images import split_image, merge_parallel, process_split_image_with_model_parallel
from datasets.datasets_pairs import my_dataset_eval

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(description="Inference: Remove shadows from input images and save outputs")
    parser.add_argument("--checkpoint", type=str, default='../autodl-tmp/SR_1/train_shadow_nafnet/nafnet_stage2_wloss.pth')
    #parser.add_argument("--checkpoint", type=str, default='./checkpoints/train_vit_stage1_wloss_total/vit_stage1_wloss.pth')
    parser.add_argument("--input_dir", type=str, default='../validation/')
    parser.add_argument("--output_dir", type=str, default='./outputs4')
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--grid_type", type=str, default="4x4")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    print("Evaluation arguments:")
    print(args)
    os.makedirs(args.output_dir, exist_ok=True)
    
    net = MaskedAutoencoderViT(
        patch_size=8, embed_dim=256, depth=6, num_heads=8,
        decoder_embed_dim=256, decoder_depth=6, decoder_num_heads=8,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6)
    )
    net.to(device)
    net.eval()
    if not os.path.exists(args.checkpoint):
        print(f"[Error] Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    net.load_state_dict(torch.load(args.checkpoint, map_location=device))
    print(f"Loaded checkpoint from {args.checkpoint}")
    
    transform = transforms.ToTensor()
    dataset = my_dataset_eval(args.input_dir, args.input_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    to_pil = ToPILImage()

    s_time = time.time()
    
    with torch.no_grad():
        for data in loader:
            if isinstance(data, (list, tuple)):
                if len(data) == 2:
                    inputs, file_names = data
                elif len(data) >= 3:
                    inputs, _, file_names = data[:3]
                else:
                    inputs = data
                    file_names = ["unknown"] * inputs.size(0)
            else:
                inputs = data
                file_names = ["unknown"] * inputs.size(0)
            inputs = inputs.to(device)
            
            orig_H, orig_W = inputs.shape[2], inputs.shape[3]
            print("Original input image size:", inputs.shape)
            
            grid_rows, grid_cols = [int(x) for x in args.grid_type.split('x')]
            
            target_H = math.ceil(orig_H / 8) * 8
            target_W = math.ceil(orig_W / 8) * 8
            
            target_H = math.ceil(target_H / grid_rows) * grid_rows
            target_W = math.ceil(target_W / grid_cols) * grid_cols

            pad_bottom = target_H - orig_H
            pad_right = target_W - orig_W
            inputs_padded = F.pad(inputs, (0, pad_right, 0, pad_bottom), mode='reflect')
            print("Padded image size:", inputs_padded.shape)
            
            sub_images, positions = split_image(inputs_padded, args.grid_type)
            sub_H = inputs_padded.shape[2] // grid_rows
            sub_W = inputs_padded.shape[3] // grid_cols
            print(f"[DEBUG] Calculated sub-image size: {sub_H}x{sub_W}")
            
            processed_sub_images = process_split_image_with_model_parallel(sub_images, net)

            outputs_padded = merge_parallel(processed_sub_images, args.grid_type)
            
            outputs = F.interpolate(outputs_padded, size=(orig_H, orig_W), mode='bilinear', align_corners=False)
            
            for i in range(outputs.size(0)):
                output_tensor = outputs[i].cpu()
                output_tensor = torch.clamp(output_tensor, 0, 1)
                output_img = to_pil(output_tensor)
                save_path = os.path.join(args.output_dir, file_names[i])
                output_img.save(save_path)
                print(f"Saved output image: {save_path}")

    e_time = time.time()
    f_time = e_time - s_time
    print(f"Elapsed time: {f_time} seconds")

if __name__ == "__main__":
    main()
