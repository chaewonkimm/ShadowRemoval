import time, argparse, logging, os, sys, gc
import torch, random
import numpy as np
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from functools import partial

from utils.UTILS import AverageMeters, print_args_parameters, compute_ssim
from utils.UTILS1 import compute_psnr
import loss.losses as losses
from torch.utils.tensorboard import SummaryWriter

import wandb

from datasets.datasets_pairs import my_dataset, my_dataset_eval
from networks.MaeVit_arch import MaskedAutoencoderViT
from networks.shadow_matte import ShadowMattePredictor
from networks.Split_images import split_image, merge, process_split_image_with_model_parallel, process_split_image_with_shadow_matte

sys.path.append(os.getcwd())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(20)

parser = argparse.ArgumentParser(description="stage1")
parser.add_argument('--experiment_name', type=str, default="vit_weight0.1_aug_sch")
parser.add_argument('--training_path', type=str, default='../aug_images/')
parser.add_argument('--max_iter', type=int, default=480000)
parser.add_argument('--img_size', type=str, default="256", help="Initial crop size as H,W or single value")
parser.add_argument('--BATCH_SIZE', type=int, default=24, help="Initial batch size")
parser.add_argument('--learning_rate', type=float, default=0.0004)
parser.add_argument('--print_frequency', type=int, default=50)
parser.add_argument('--fft_loss_weight', type=float, default=0.1, help="Weight for FFT loss")
parser.add_argument('--grid_type', type=str, default="4x4", help="Grid type for dynamic splitting")
parser.add_argument('--val_interval', type=int, default=5000, help="Interval for validation")
parser.add_argument('--checkpoint_path', type=str, help="checkpoints", default=None) # pretrained 여부
parser.add_argument('--resume_iter', type=int, default=0, help="Iteration from which to resume training")
parser.add_argument('--shadow_matte_path', type=str, default="checkpoints/shadow_matte/epoch_1400.pth", help="Path to the shadow matte model weights")
args = parser.parse_args()

print_args_parameters(args)

if ',' in args.img_size:
    current_img_size = tuple(map(int, args.img_size.split(',')))
else:
    current_img_size = int(args.img_size)

wandb.init(project="shadow_removal", name=args.experiment_name, config=vars(args))
SAVE_PATH = os.path.join('./checkpoints', args.experiment_name)
os.makedirs(SAVE_PATH, exist_ok=True)
logging.basicConfig(filename=os.path.join(SAVE_PATH, f"{args.experiment_name}.log"), level=logging.INFO)

def get_dataset(img_size):
    return my_dataset(root_dir=args.training_path, crop_size=img_size, fix_sample_A=999, regular_aug=False)

def get_dataloaders(img_size):
    dataset = get_dataset(img_size)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=args.BATCH_SIZE, num_workers=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.BATCH_SIZE, num_workers=8, shuffle=False)
    return train_loader, val_loader

train_loader, val_loader = get_dataloaders(current_img_size)

matte_predictor = ShadowMattePredictor(
    model_path=args.shadow_matte_path,
    img_size=int(args.img_size) if isinstance(args.img_size, str) and not ',' in args.img_size else 256,
    device=device
)

net = MaskedAutoencoderViT(
    patch_size=8, embed_dim=256, depth=6, num_heads=8,
    decoder_embed_dim=256, decoder_depth=6, decoder_num_heads=8,
    mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6)
)
net.to(device)
print('#parameters:', sum(p.numel() for p in net.parameters()))
logging.info(f"#parameters: {sum(p.numel() for p in net.parameters())}")

optimizer = optim.Adam(net.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))
for param_group in optimizer.param_groups:
    param_group['lr'] = args.learning_rate
scheduler = CosineAnnealingLR(optimizer, T_max=args.max_iter, eta_min=8e-5)

base_loss = losses.CharbonnierLoss()
fft_loss_fn = losses.fftLoss()

global_iter = 0

if args.checkpoint_path and os.path.exists(args.checkpoint_path):
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    net.load_state_dict(checkpoint)
    global_iter = args.resume_iter
    print(f"Checkpoint loaded from {args.checkpoint_path}, resuming from iteration {global_iter}")
    logging.info(f"Checkpoint loaded from {args.checkpoint_path}, resuming from iteration {global_iter}")
else:
    print("No checkpoint loaded. Starting training from scratch.")

max_iter = args.max_iter
train_iter = iter(train_loader)

while global_iter < max_iter:
    try:
        data_in, label, img_name = next(train_iter)
    except StopIteration:
        train_iter = iter(train_loader)
        data_in, label, img_name = next(train_iter)

    optimizer.zero_grad()
    inputs = data_in.to(device)
    labels = label.to(device)

    with torch.no_grad():
        shadow_matte = matte_predictor.predict_from_tensor(inputs) # [B, 1, img_size, img_size]

    sub_images, positions = split_image(inputs, args.grid_type)
    sub_mattes, _ = split_image(shadow_matte, args.grid_type) # [384(B * 16), 1, 64, 64]

    processed_sub_images = process_split_image_with_shadow_matte(sub_images, sub_mattes, net)
    # processed_sub_images = process_split_image_with_model_parallel(sub_images, net)
    outputs = merge(processed_sub_images, positions)
    loss_char = base_loss(outputs, labels)
    loss_fft = fft_loss_fn(outputs, labels)
    loss = loss_char + args.fft_loss_weight * loss_fft
    loss.backward()
    optimizer.step()
    global_iter += 1

    if global_iter % args.print_frequency == 0:
        psnr_val = compute_psnr(outputs, labels)
        ssim_val = compute_ssim(outputs, labels)
        print(f"Iter {global_iter} | CharLoss: {loss_char.item():.4f}, FFTLoss: {loss_fft.item():.4f}, TotalLoss: {loss.item():.4f}, PSNR: {psnr_val:.2f} dB, SSIM: {ssim_val:.4f}")
        logging.info(f"Iter {global_iter} | CharLoss: {loss_char.item():.4f}, FFTLoss: {loss_fft.item():.4f}, TotalLoss: {loss.item():.4f}, PSNR: {psnr_val:.2f} dB, SSIM: {ssim_val:.4f}")
        wandb.log({
            "iter_loss": loss.item(),
            "iter_char_loss": loss_char.item(),
            "iter_fft_loss": loss_fft.item(),
            "iter_psnr": psnr_val,
            "iter_ssim": ssim_val,
            "global_iter": global_iter
        })

    if global_iter % args.val_interval == 0:
        net.eval()
        val_total_loss = 0.0
        val_total_psnr = 0.0
        val_total_ssim = 0.0
        val_count = 0
        with torch.no_grad():
            for data_in, label, img_name in val_loader:
                inputs = data_in.to(device)
                labels = label.to(device)

                shadow_matte = matte_predictor.predict_from_tensor(inputs)

                sub_images, positions = split_image(inputs, args.grid_type)
                sub_mattes, _ = split_image(shadow_matte, args.grid_type)
                processed_sub_images = process_split_image_with_shadow_matte(sub_images, sub_mattes, net)
                # processed_sub_images = process_split_image_with_model_parallel(sub_images, net)
                
                outputs = merge(processed_sub_images, positions)
                loss_char = base_loss(outputs, labels)
                loss_fft = fft_loss_fn(outputs, labels)
                loss = loss_char + args.fft_loss_weight * loss_fft
                val_total_loss += loss.item()
                psnr_val = compute_psnr(outputs, labels)
                ssim_val = compute_ssim(outputs, labels)
                val_total_psnr += psnr_val
                val_total_ssim += ssim_val
                val_count += 1
        avg_val_loss = val_total_loss / val_count if val_count > 0 else 0
        avg_val_psnr = val_total_psnr / val_count if val_count > 0 else 0
        avg_val_ssim = val_total_ssim / val_count if val_count > 0 else 0
        print(f"[Validation] Iter {global_iter} | Loss: {avg_val_loss:.4f}, PSNR: {avg_val_psnr:.2f} dB, SSIM: {avg_val_ssim:.4f}")
        logging.info(f"[Validation] Iter {global_iter} | Loss: {avg_val_loss:.4f}, PSNR: {avg_val_psnr:.2f} dB, SSIM: {avg_val_ssim:.4f}")
        wandb.log({
            "val_loss": avg_val_loss,
            "val_psnr": avg_val_psnr,
            "val_ssim": avg_val_ssim,
            "global_iter": global_iter
        })
        checkpoint_path = os.path.join(SAVE_PATH, f"vit_stage1_iter_{global_iter}.pth")
        torch.save(net.state_dict(), checkpoint_path)
        print(f"Checkpoint saved at iteration {global_iter}")
        logging.info(f"Checkpoint saved at iteration {global_iter}")

    if scheduler is not None:
        scheduler.step()

torch.save(net.state_dict(), os.path.join(SAVE_PATH, "vit_stage1_weight.pth"))
print("Training complete: ViT model saved.")
logging.info("Training complete: ViT model saved.")
wandb.finish()