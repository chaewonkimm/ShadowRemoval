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
from networks.Split_images import split_image, merge, process_split_image_with_model_parallel

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
parser.add_argument('--experiment_name', type=str, default="train_vit_stage1")
parser.add_argument('--training_path', type=str, default='../train/')
parser.add_argument('--max_iter', type=int, default=124000)
parser.add_argument('--img_size', type=str, default="256", help="Initial crop size as H,W or single value")
parser.add_argument('--BATCH_SIZE', type=int, default=24, help="Initial batch size")
parser.add_argument('--learning_rate', type=float, default=0.0004)
parser.add_argument('--print_frequency', type=int, default=50)
parser.add_argument('--fft_loss_weight', type=float, default=0.1, help="Weight for FFT loss")
parser.add_argument('--grid_type', type=str, default="4x4", help="Grid type for dynamic splitting")
parser.add_argument('--val_interval', type=int, default=5000, help="Interval for validation")
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

progressive_schedule = [
    (40000, (512, 512), 12),
    (76000, (1024, 1024), 5),
    (100000, (1408, 1408), 3)
]
current_schedule_index = 0
phase4_started = False

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
scheduler = None

base_loss = losses.CharbonnierLoss()
fft_loss_fn = losses.fftLoss()
global_iter = 0
max_iter = args.max_iter
train_iter = iter(train_loader)

while global_iter < max_iter:
    try:
        data_in, label, img_name = next(train_iter)
    except StopIteration:
        train_iter = iter(train_loader)
        data_in, label, img_name = next(train_iter)

    if current_schedule_index < len(progressive_schedule) and global_iter >= progressive_schedule[current_schedule_index][0]:
        new_img_size, new_batch_size = progressive_schedule[current_schedule_index][1], progressive_schedule[current_schedule_index][2]
        current_img_size = new_img_size
        args.BATCH_SIZE = new_batch_size
        train_loader, val_loader = get_dataloaders(current_img_size)
        train_iter = iter(train_loader)
        print(f"Progressive update at iter {global_iter}: img_size -> {current_img_size}, batch_size -> {new_batch_size}")
        logging.info(f"Progressive update at iter {global_iter}: img_size -> {current_img_size}, batch_size -> {new_batch_size}")
        if current_img_size == (1408, 1408) and not phase4_started:
            remaining_iter = max_iter - global_iter
            scheduler = CosineAnnealingLR(optimizer, T_max=remaining_iter, eta_min=8e-5)
            phase4_started = True
        current_schedule_index += 1

    optimizer.zero_grad()
    inputs = data_in.to(device)
    labels = label.to(device)
    sub_images, positions = split_image(inputs, args.grid_type)
    processed_sub_images = process_split_image_with_model_parallel(sub_images, net)
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
                sub_images, positions = split_image(inputs, args.grid_type)
                processed_sub_images = process_split_image_with_model_parallel(sub_images, net)
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

    if phase4_started and scheduler is not None:
        scheduler.step()

torch.save(net.state_dict(), os.path.join(SAVE_PATH, "vit_stage1.pth"))
print("Training complete: ViT model saved.")
logging.info("Training complete: ViT model saved.")
wandb.finish()
