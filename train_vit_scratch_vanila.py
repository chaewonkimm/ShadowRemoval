import time, argparse, logging, os, sys, gc
import torch, random
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from functools import partial

from utils.UTILS import AverageMeters, print_args_parameters
from utils.UTILS1 import compute_psnr
import loss.losses as losses

from datasets.datasets_pairs import my_dataset, my_dataset_eval

from networks.MaeVit_arch import MaskedAutoencoderViT

sys.path.append(os.getcwd())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(20)

parser = argparse.ArgumentParser(description="Stage 1: Train ViT with Charbonnier + FFT Loss")
parser.add_argument('--experiment_name', type=str, default="train_vit_stage1")
parser.add_argument('--training_path', type=str, default='../train/')
parser.add_argument('--EPOCH', type=int, default=600)
parser.add_argument('--BATCH_SIZE', type=int, default=24)
parser.add_argument('--learning_rate', type=float, default=0.0004)
parser.add_argument('--print_frequency', type=int, default=50)
parser.add_argument('--Crop_patches', type=int, default=224)
parser.add_argument('--fft_loss_weight', type=float, default=0.02, 
                    help="Weight for FFT loss (e.g., 0.02 ~ 0.1)")
args = parser.parse_args()

print_args_parameters(args)

SAVE_PATH = os.path.join('./checkpoints', args.experiment_name)
os.makedirs(SAVE_PATH, exist_ok=True)
logging.basicConfig(filename=os.path.join(SAVE_PATH, f"{args.experiment_name}.log"), level=logging.INFO)

def get_training_data(crop_size=args.Crop_patches):
    train_dataset = my_dataset(
        root_dir=args.training_path,
        crop_size=crop_size,
        fix_sample_A=999,
        regular_aug=False
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.BATCH_SIZE,
        num_workers=8,
        shuffle=True
    )
    return train_loader

train_loader = get_training_data()

net = MaskedAutoencoderViT(
    patch_size=8, embed_dim=256, depth=6, num_heads=8,
    decoder_embed_dim=256, decoder_depth=6, decoder_num_heads=8,
    mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6)
)
net.to(device)

print('#parameters:', sum(p.numel() for p in net.parameters()))
logging.info(f"#parameters: {sum(p.numel() for p in net.parameters())}")

optimizer = optim.Adam(net.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=1)

base_loss = losses.CharbonnierLoss()

fft_loss_fn = losses.fftLoss()

for epoch in range(args.EPOCH):
    scheduler.step(epoch)
    net.train()
    epoch_loss = 0.0

    for i, (data_in, label, img_name) in enumerate(train_loader):
        optimizer.zero_grad()

        inputs = data_in.to(device)
        labels = label.to(device)

        outputs = net(inputs)

        loss_char = base_loss(outputs, labels)

        loss_fft = fft_loss_fn(outputs, labels)

        loss = loss_char + args.fft_loss_weight * loss_fft

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        if (i+1) % args.print_frequency == 0:
            print(f"[Stage 1][Epoch {epoch}][Batch {i+1}/{len(train_loader)}] "
                  f"CharLoss: {loss_char.item():.4f}, FFTLoss: {loss_fft.item():.4f}, "
                  f"TotalLoss: {loss.item():.4f}")
            logging.info(f"[Stage 1][Epoch {epoch}][Batch {i+1}/{len(train_loader)}] "
                         f"CharLoss: {loss_char.item():.4f}, FFTLoss: {loss_fft.item():.4f}, "
                         f"TotalLoss: {loss.item():.4f}")

    avg_loss = epoch_loss / len(train_loader)
    print(f"[Stage 1][Epoch {epoch}] Average Loss: {avg_loss:.4f}")
    logging.info(f"[Stage 1][Epoch {epoch}] Average Loss: {avg_loss:.4f}")

torch.save(net.state_dict(), os.path.join(SAVE_PATH, "vit_stage1.pth"))
print("Stage 1 complete: ViT model saved with Charbonnier + FFT Loss.")
logging.info("Stage 1 complete: ViT model saved.")
