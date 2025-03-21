import time, torchvision, argparse, logging, sys, os, gc
import torch, random, math
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from utils.UTILS1 import compute_psnr
from utils.UTILS import AverageMeters, print_args_parameters, Lion, compute_ssim
import loss.losses as losses
from torch.utils.tensorboard import SummaryWriter
from functools import partial
from datasets.datasets_pairs import my_dataset, my_dataset_eval
from networks.NAFNet_arch import NAFNet_CBAM
from networks.MaeVit_arch import MaskedAutoencoderViT
from networks.shadow_matte import ShadowMattePredictor
from networks.Split_images import split_image, merge, process_split_image_with_model_parallel, process_split_image_with_shadow_matte
from PIL import Image
import wandb
sys.path.append(os.getcwd())

def get_high_freq_weight_map(target, device, kernel_size=3, padding=1):
    laplacian_kernel = torch.tensor([[0, -1, 0],
                                     [-1, 4, -1],
                                     [0, -1, 0]], dtype=torch.float32, device=device)
    laplacian_kernel = laplacian_kernel.unsqueeze(0).unsqueeze(0)
    C = target.size(1)
    laplacian_kernel = laplacian_kernel.repeat(C, 1, 1, 1)
    high_freq = nn.functional.conv2d(target, laplacian_kernel, groups=C, padding=padding)
    weight_map = torch.abs(high_freq)
    weight_map = (weight_map - weight_map.min()) / (weight_map.max() - weight_map.min() + 1e-8)
    return weight_map

def weighted_charbonnier_loss(output, target, epsilon=1e-3, weight_coff=1):
    device = output.device
    weight_map = get_high_freq_weight_map(target, device)
    loss = torch.sqrt(weight_coff * weight_map * (output - target)**2 + epsilon**2)
    return loss.mean()

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

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(20)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device ----------------------------------------:', device)

def get_full_dataset():
    return my_dataset(root_dir=args.training_path, crop_size=args.img_size, fix_sample_A=fix_sampleA, regular_aug=args.Aug_regular)

parser = argparse.ArgumentParser(description="ViT & NAFNet finetuning pipeline with matte predictor and sliding crop")
parser.add_argument('--vit_patch_size', type=int, default=8)
parser.add_argument('--vit_embed_dim', type=int, default=256)
parser.add_argument('--vit_depth', type=int, default=6)
parser.add_argument('--vit_num_heads', type=int, default=8)
parser.add_argument('--vit_decoder_embed_dim', type=int, default=256)
parser.add_argument('--vit_decoder_depth', type=int, default=6)
parser.add_argument('--vit_decoder_num_heads', type=int, default=8)
parser.add_argument('--vit_mlp_ratio', type=int, default=4)
parser.add_argument('--vit_img_size', type=int, default=256)  # ViT input size with matte
parser.add_argument('--img_size', type=int, default=750)  # sliding crop size (750x750)
parser.add_argument('--grid_type', type=str, default="4x4", help="Grid type for dynamic splitting (NAFNet)")
parser.add_argument('--overlap_size', type=int, default=16)  # overlap for 256x256 patch split
parser.add_argument('--crop_stride', type=int, default=250, help="Stride for sliding crop from left")
parser.add_argument('--Flag_process_split_image_with_model_parallel', type=bool, default=True)
parser.add_argument('--Flag_multi_scale', type=bool, default=False)
parser.add_argument('--experiment_name', type=str, default="finetune_vit_wnafnet_matte0.1_CBAM")
parser.add_argument('--unified_path', type=str, default='./autodl-tmp/SR_1/')
parser.add_argument('--T_period', type=int, default=50)
parser.add_argument('--training_path', type=str, default='../aug_images', help='Training images folder')
parser.add_argument('--writer_dir', type=str, default='./tf-logs/')
parser.add_argument('--infer_path', type=str, default='./test/input', help='Inference input images folder')
parser.add_argument('--iteration_target', type=int, default=80000)
parser.add_argument('--BATCH_SIZE', type=int, default=1)
parser.add_argument('--Crop_patches', type=int, default=0)
parser.add_argument('--nafnet_learning_rate', type=float, default=2e-4)
parser.add_argument('--vit_learning_rate', type=float, default=8e-5)
parser.add_argument('--print_frequency', type=int, default=50)
parser.add_argument('--fft_loss_weight', type=float, default=0.1, help="Weight for FFT loss")
parser.add_argument('--SAVE_Inter_Results', type=bool, default=False)
parser.add_argument('--fix_sampleA', type=int, default=999)
parser.add_argument('--debug', type=bool, default=False)
parser.add_argument('--Aug_regular', type=bool, default=False)
parser.add_argument('--base_channel', type=int, default=32)
parser.add_argument('--num_res', type=int, default=24)
parser.add_argument('--img_channel', type=int, default=3)
parser.add_argument('--enc_blks', nargs='+', type=int, default=[1, 1, 1, 28], help='List of integers')
parser.add_argument('--dec_blks', nargs='+', type=int, default=[1, 1, 1, 1], help='List of integers')
parser.add_argument('--base_loss', type=str, default='weightedchar')
parser.add_argument('--addition_loss', type=str, default='None')
parser.add_argument('--addition_loss_coff', type=float, default=0.02)
parser.add_argument('--weight_coff', type=float, default=10.0)
parser.add_argument('--load_pre_model', type=bool, default=False)
parser.add_argument('--pre_model', type=str, default='./pretrained/vit_matte_weight0.1_finetuning.pth')
parser.add_argument('--pre_model_0', type=str, default='./pretrained/vit_matte_weight0.1_finetuning.pth')
parser.add_argument('--pre_model_1', type=str, default='../autodl-tmp/SR_1/train_nafnet_wmatte_matte_depth/checkpoint_iter_40000.pth')
parser.add_argument('--shadow_matte_path', type=str, default="pretrained/epoch_1400.pth")
parser.add_argument('--optim', type=str, default='adam')
args = parser.parse_args()
print_args_parameters(args)

if args.debug:
    fix_sampleA = 400
else:
    fix_sampleA = args.fix_sampleA

exper_name = args.experiment_name
writer = SummaryWriter(args.writer_dir + exper_name)
if not os.path.exists(args.writer_dir):
    os.makedirs(args.writer_dir, exist_ok=True)
unified_path = args.unified_path
SAVE_PATH = os.path.join(unified_path, exper_name) + '/'
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH, exist_ok=True)
if args.SAVE_Inter_Results:
    SAVE_Inter_Results_PATH = os.path.join(SAVE_PATH, 'Inter_Temp_results/')
    if not os.path.exists(SAVE_Inter_Results_PATH):
        os.makedirs(SAVE_Inter_Results_PATH, exist_ok=True)

logging.basicConfig(filename=os.path.join(SAVE_PATH, f"{args.experiment_name}.log"), level=logging.INFO)
for k in args.__dict__:
    logging.info(k + ": " + str(args.__dict__[k]))
logging.info('begin training!')
print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def get_full_dataset():
    return my_dataset(root_dir=args.training_path, crop_size=args.img_size, fix_sample_A=fix_sampleA, regular_aug=args.Aug_regular)

full_dataset = get_full_dataset()
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
train_loader = DataLoader(dataset=train_dataset, batch_size=args.BATCH_SIZE, num_workers=8, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=1, num_workers=4, shuffle=False)
print('Train dataset size:', len(train_dataset))
print('Validation dataset size:', len(val_dataset))
logging.info('Train dataset size: %d', len(train_dataset))
logging.info('Validation dataset size: %d', len(val_dataset))

class InferenceDataset(Dataset):
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

def get_inference_data(infer_path=args.infer_path):
    transform = transforms.Compose([transforms.ToTensor()])
    infer_dataset = InferenceDataset(infer_path, transform)
    infer_loader = DataLoader(dataset=infer_dataset, batch_size=1, num_workers=4)
    print('len(infer_loader):', len(infer_loader))
    logging.info('len(infer_loader): %d', len(infer_loader))
    return infer_loader

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

def validate(net, net_1, val_loader, val_save_dir, iteration):
    net.eval()
    net_1.eval()
    total_psnr = 0
    total_ssim = 0
    count = 0
    os.makedirs(val_save_dir, exist_ok=True)
    with torch.no_grad():
        for data, label, fname in val_loader:
            inputs = data.to(device)
            gt = label.to(device)
            B, C, H, W = inputs.shape
            crops, crop_positions = sliding_crop_left(inputs, patch_size=args.img_size, stride=args.crop_stride)
            processed_crops = []
            processed_mattes = []
            # 각 crop patch에 ViT 적용
            for crop in crops:
                sub_data, positions = split_image_overlap(crop, crop_size=256, overlap_size=args.overlap_size)
                matte_patches = []
                for sub in sub_data:
                    with torch.no_grad():
                        matte_patch = matte_predictor.predict_from_tensor(sub)
                    matte_patches.append(matte_patch)
                processed = process_split_image_with_shadow_matte(sub_data, matte_patches, net)
                crop_processed = merge_image_overlap(processed, positions, crop_size=256, resolution=crop.shape, overlap_size=args.overlap_size, blend_mode='gaussian')
                crop_mattes = merge_image_overlap(matte_patches, positions, crop_size=256, resolution=(args.BATCH_SIZE, 1, args.img_size, args.img_size), overlap_size=args.overlap_size, blend_mode='gaussian')
                processed_crops.append(crop_processed)
                processed_mattes.append(crop_mattes)
            outputs = merge_sliding_crops(processed_crops, crop_positions, original_width=W, overlap_size=args.overlap_size)
            mattes = merge_sliding_crops(processed_mattes, crop_positions, original_width=inputs.shape[-1], overlap_size=args.overlap_size)
            nafnet_output = net_1(outputs, mattes)
            psnr_val = compute_psnr(nafnet_output, gt)
            ssim_val = compute_ssim(nafnet_output, gt)
            total_psnr += psnr_val
            total_ssim += ssim_val
            count += 1
            save_path = os.path.join(val_save_dir, f"{fname[0]}_iter{iteration}.png")
            torchvision.utils.save_image(nafnet_output.cpu()[0], save_path)
        avg_psnr = total_psnr / count if count > 0 else 0
        avg_ssim = total_ssim / count if count > 0 else 0
    return avg_psnr, avg_ssim

def inference(net, net_1, infer_loader, save_dir):
    net.eval()
    net_1.eval()
    os.makedirs(save_dir, exist_ok=True)
    with torch.no_grad():
        for idx, (data, fname) in enumerate(infer_loader):
            inputs = data.to(device)
            B, C, H, W = inputs.shape
            crops, crop_positions = sliding_crop_left(inputs, patch_size=args.img_size, stride=args.crop_stride)
            processed_crops = []
            processed_mattes = []
            for crop in crops:
                sub_data, positions = split_image_overlap(crop, crop_size=256, overlap_size=args.overlap_size)
                matte_patches = []
                for sub in sub_data:
                    with torch.no_grad():
                        matte_patch = matte_predictor.predict_from_tensor(sub)
                    matte_patches.append(matte_patch)
                processed = process_split_image_with_shadow_matte(sub_data, matte_patches, net)
                crop_processed = merge_image_overlap(processed, positions, crop_size=256, resolution=crop.shape, overlap_size=args.overlap_size, blend_mode='gaussian')
                crop_mattes = merge_image_overlap(matte_patches, positions, crop_size=256, resolution=(args.BATCH_SIZE, 1, args.img_size, args.img_size), overlap_size=args.overlap_size, blend_mode='gaussian')
                processed_crops.append(crop_processed)
                processed_mattes.append(crop_mattes)
            outputs = merge_sliding_crops(processed_crops, crop_positions, original_width=W, overlap_size=args.overlap_size)
            mattes = merge_sliding_crops(processed_mattes, crop_positions, original_width=W, overlap_size=args.overlap_size)
            print(f'mattes: {mattes.shape}')
            nafnet_output = net_1(outputs, shadow_mate=mattes)
            save_path = os.path.join(save_dir, fname[0])
            torchvision.utils.save_image(nafnet_output.cpu()[0], save_path)
            print(f"Saved inference output: {save_path}")

def print_param_number(net):
    total_params = sum(param.numel() for param in net.parameters())
    print('#generator parameters:', total_params)
    logging.info('#generator parameters: %d', total_params)

VAL_SAVE_DIR = os.path.join(SAVE_PATH, "validation_results")

if __name__ == '__main__':
    wandb.init(project="nafnet_wloss", name=args.experiment_name, config=vars(args))
    if args.Flag_multi_scale:
        net_1 = NAFNet_CBAM(img_channel=args.img_channel, width=args.base_channel, middle_blk_num=args.num_res,
                       enc_blk_nums=args.enc_blks, dec_blk_nums=args.dec_blks, global_residual=False)
    else:
        net_1 = NAFNet_CBAM(img_channel=args.img_channel, width=args.base_channel, middle_blk_num=args.num_res,
                       enc_blk_nums=args.enc_blks, dec_blk_nums=args.dec_blks, global_residual=False)
    net = MaskedAutoencoderViT(patch_size=args.vit_patch_size, embed_dim=args.vit_embed_dim, depth=args.vit_depth,
                              num_heads=args.vit_num_heads, decoder_embed_dim=args.vit_decoder_embed_dim,
                              decoder_depth=args.vit_decoder_depth, decoder_num_heads=args.vit_decoder_num_heads,
                              mlp_ratio=args.vit_mlp_ratio, norm_layer=partial(nn.LayerNorm, eps=1e-6))
    matte_predictor = ShadowMattePredictor(
        model_path=args.shadow_matte_path,
        img_size=args.vit_img_size,
        device=device
    )
    net.load_state_dict(torch.load(args.pre_model), strict=True)
    print('-----'*20, 'successfully load vit-pre-trained weights!!!!!')
    if args.load_pre_model:
        net.load_state_dict(torch.load(args.pre_model_0), strict=True)
        print('-----'*20, 'successfully load pre-trained weights!!!!!')
        logging.info('-----'*20, 'successfully load pre-trained weights!!!!!')
        net_1.load_state_dict(torch.load(args.pre_model_1), strict=True)
        print('-----'*20, 'successfully load pre-trained weights!!!!!')
        logging.info('-----'*20, 'successfully load pre-trained weights!!!!!')
    net.to(device)
    print_param_number(net)
    net_1.to(device)
    print_param_number(net_1)
    wandb.watch(net_1, log="all")

    optimizer_vit = optim.Adam(net.parameters(), lr=args.vit_learning_rate, betas=(0.9, 0.999))
    optimizer_nafnet = optim.Adam(net_1.parameters(), lr=args.nafnet_learning_rate, betas=(0.9, 0.999))
    scheduler_vit = CosineAnnealingWarmRestarts(optimizer=optimizer_vit, T_0=args.T_period, T_mult=1)
    scheduler_nafnet = CosineAnnealingWarmRestarts(optimizer=optimizer_nafnet, T_0=args.T_period, T_mult=1)

    base_loss = weighted_charbonnier_loss if args.base_loss.lower() == 'weightedchar' else nn.L1Loss()
    if args.addition_loss.lower() == 'vgg':
        criterion = losses.VGGLoss()
    elif args.addition_loss.lower() == 'ssim':
        criterion = losses.SSIMLoss()
    else:
        criterion = None
    running_results = {'iter_nums': 0}
    Avg_Meters_training = AverageMeters()
    global_iter = 0
    while global_iter < args.iteration_target:
        for i, train_data in enumerate(train_loader, 0):
            data_in, label, img_name = train_data
            if i == 0:
                print(f" train_input.size: {data_in.size()}, gt.size: {label.size()}")
            running_results['iter_nums'] += 1
            net_1.train()
            net.train()  # Ensure ViT is in train mode too.
            optimizer_vit.zero_grad()
            optimizer_nafnet.zero_grad()

            inputs = data_in.to(device)
            labels = label.to(device)
            crops, crop_positions = sliding_crop_left(inputs, patch_size=args.img_size, stride=args.crop_stride)
            processed_crops = []
            processed_mattes = []
            for crop in crops:
                sub_data, positions = split_image_overlap(crop, crop_size=256, overlap_size=args.overlap_size)
                matte_patches = []
                for sub in sub_data:
                    with torch.no_grad():
                        matte_patch = matte_predictor.predict_from_tensor(sub)
                    matte_patches.append(matte_patch)
                processed = process_split_image_with_shadow_matte(sub_data, matte_patches, net)
                crop_processed = merge_image_overlap(processed, positions, crop_size=256, resolution=crop.shape, overlap_size=args.overlap_size, blend_mode='gaussian')
                crop_mattes = merge_image_overlap(matte_patches, positions, crop_size=256, resolution=(args.BATCH_SIZE, 1, args.img_size, args.img_size), overlap_size=args.overlap_size, blend_mode='gaussian')
                processed_crops.append(crop_processed)
                processed_mattes.append(crop_mattes)
            outputs = merge_sliding_crops(processed_crops, crop_positions, original_width=inputs.shape[-1], overlap_size=args.overlap_size)
            mattes = merge_sliding_crops(processed_mattes, crop_positions, original_width=inputs.shape[-1], overlap_size=args.overlap_size)
            nafnet_output = net_1(outputs, mattes)
            loss1 = base_loss(nafnet_output, labels, args.weight_coff)
            fft_loss = losses.fftLoss()(nafnet_output, labels)
            loss_total = loss1 + args.fft_loss_weight * fft_loss
            loss_addition = args.addition_loss_coff * criterion(nafnet_output, labels) if criterion is not None else 0
            loss_total = loss_total + loss_addition
            Avg_Meters_training.update({'total_loss': loss_total.item()})
            loss_total.backward()
            optimizer_vit.step()
            optimizer_nafnet.step()
            global_iter += 1

            if global_iter % 5000 == 0:
                checkpoint_path = os.path.join(SAVE_PATH, f"vit_{global_iter}.pth")
                checkpoint_path1 = os.path.join(SAVE_PATH, f"nafnet_{global_iter}.pth")
                torch.save(net.state_dict(), checkpoint_path)
                torch.save(net_1.state_dict(), checkpoint_path1)
                print(f"Checkpoint saved at iteration {global_iter}")
                logging.info(f"Checkpoint saved at iteration {global_iter}")
            
            if (i + 1) % args.print_frequency == 0 and i > 1:
                psnr_val = compute_psnr(nafnet_output, labels)
                ssim_val = compute_ssim(nafnet_output, labels)
                print("Iteration:%d, [vit_lr: %.7f], [nafnet_lr: %.7f], [loss_total: %.5f], PSNR: %.2f, SSIM: %.4f" %
                        (global_iter, optimizer_vit.param_groups[0]["lr"], optimizer_nafnet.param_groups[0]["lr"], loss_total.item(), psnr_val, ssim_val))
                wandb.log({
                    "iteration": global_iter,
                    "loss_total": loss_total.item(),
                    "loss_char": loss1.item(),
                    "loss_fft": fft_loss.item(),
                    "iter_psnr": psnr_val,
                    "iter_ssim": ssim_val,
                    "vit_lr": optimizer_vit.param_groups[0]["lr"],
                    "nafnet_lr": optimizer_nafnet.param_groups[0]["lr"]
                })
            if global_iter % 5000 == 0:
                val_psnr, val_ssim = validate(net, net_1, val_loader, VAL_SAVE_DIR, global_iter)
                print(f"Validation at Iteration {global_iter}: PSNR = {val_psnr:.2f}, SSIM = {val_ssim:.4f}")
                wandb.log({
                    "validation_iter": global_iter,
                    "val_psnr": val_psnr,
                    "val_ssim": val_ssim
                })
            if global_iter >= args.iteration_target:
                break
        scheduler_vit.step()
        scheduler_nafnet.step()
    torch.save(net.state_dict(), os.path.join(SAVE_PATH, "train_vit_wmatte"))
    torch.save(net_1.state_dict(), os.path.join(SAVE_PATH, "train_nafnet_wmatte"))
    print("Training complete: NAFNet model (finetuned) saved.")
    logging.info("Training complete: NAFNet model (finetuned) saved.")
    wandb.finish()