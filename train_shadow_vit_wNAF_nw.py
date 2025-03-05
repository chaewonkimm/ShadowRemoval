import time, torchvision, argparse, logging, sys, os, gc
import torch, random, math
import numpy as np
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from utils.UTILS1 import compute_psnr
from utils.UTILS import AverageMeters, print_args_parameters, Lion, compute_ssim
import loss.losses as losses
from torch.utils.tensorboard import SummaryWriter
from functools import partial
from datasets.datasets_pairs import my_dataset
from networks.NAFNet_arch import NAFNet
from networks.MaeVit_arch import MaskedAutoencoderViT
from networks.image_utils import splitimage, mergeimage
from PIL import Image
import wandb
sys.path.append(os.getcwd())

def split_image_overlap(img, crop_size, overlap_size):
    B, C, H, W = img.shape
    stride = crop_size - overlap_size
    y_starts = list(range(0, H - crop_size + 1, stride))
    if y_starts[-1] != H - crop_size:
        y_starts.append(H - crop_size)
    x_starts = list(range(0, W - crop_size + 1, stride))
    if x_starts[-1] != W - crop_size:
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

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(20)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device ----------------------------------------:', device)

parser = argparse.ArgumentParser()
parser.add_argument('--vit_patch_size', type=int, default=8)
parser.add_argument('--vit_embed_dim', type=int, default=256)
parser.add_argument('--vit_depth', type=int, default=6)
parser.add_argument('--vit_num_heads', type=int, default=8)
parser.add_argument('--vit_decoder_embed_dim', type=int, default=256)
parser.add_argument('--vit_decoder_depth', type=int, default=6)
parser.add_argument('--vit_decoder_num_heads', type=int, default=8)
parser.add_argument('--vit_mlp_ratio', type=int, default=4)
parser.add_argument('--vit_img_size', type=int, default=1024)
parser.add_argument('--vit_grid_type', type=str, default='4x4')
parser.add_argument('--Flag_process_split_image_with_model_parallel', type=bool, default=True)
parser.add_argument('--Flag_multi_scale', type=bool, default=False)
parser.add_argument('--experiment_name', type=str, default="train_shadow_nafnet_32_overlap")
parser.add_argument('--unified_path', type=str, default='/root/autodl-tmp/SR_1/')
parser.add_argument('--T_period', type=int, default=50)
parser.add_argument('--training_path', type=str, default='../train', help='Training images folder')
parser.add_argument('--writer_dir', type=str, default='/root/tf-logs/')
parser.add_argument('--infer_path', type=str, default='./test/input', help='Inference input images folder')
parser.add_argument('--iteration_target', type=int, default=40000)
parser.add_argument('--BATCH_SIZE', type=int, default=1)
parser.add_argument('--overlap_size', type=int, default=32)
parser.add_argument('--Crop_patches', type=int, default=1024)
parser.add_argument('--learning_rate', type=float, default=8e-5)
parser.add_argument('--print_frequency', type=int, default=50)
parser.add_argument('--fft_loss_weight', type=float, default=0.1, help="Weight for FFT loss")
parser.add_argument('--SAVE_Inter_Results', type=bool, default=False)
parser.add_argument('--fix_sampleA', type=int, default=999)
parser.add_argument('--debug', type=bool, default=False)
parser.add_argument('--Aug_regular', type=bool, default=False)
parser.add_argument('--base_channel', type=int, default=24)
parser.add_argument('--num_res', type=int, default=6)
parser.add_argument('--img_channel', type=int, default=3)
parser.add_argument('--enc_blks', nargs='+', type=int, default=[1, 1, 1, 28], help='List of integers')
parser.add_argument('--dec_blks', nargs='+', type=int, default=[1, 1, 1, 1], help='List of integers')
parser.add_argument('--base_loss', type=str, default='char')
parser.add_argument('--addition_loss', type=str, default='None')
parser.add_argument('--addition_loss_coff', type=float, default=0.02)
parser.add_argument('--weight_coff', type=float, default=10.0)
parser.add_argument('--load_pre_model', type=bool, default=False)
parser.add_argument('--pre_model', type=str, default='./checkpoints/train_vit_stage1_wloss_total/vit_stage1_wloss.pth')
parser.add_argument('--pre_model_0', type=str, default='./checkpoints/train_vit_stage1_wloss_total/vit_stage1_wloss.pth')
parser.add_argument('--pre_model_1', type=str, default='')
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

logging.basicConfig(filename=os.path.join(SAVE_PATH, args.experiment_name + '.log'), level=logging.INFO)
for k in args.__dict__:
    logging.info(k + ": " + str(args.__dict__[k]))
logging.info('begin training!')
print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def get_full_dataset():
    return my_dataset(root_dir=args.training_path, crop_size=args.Crop_patches, fix_sample_A=fix_sampleA, regular_aug=args.Aug_regular)

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
        _, h, w = img.shape
        if (h % 16 != 0) or (w % 16 != 0):
            img = transforms.Resize(((h // 16) * 16, (w // 16) * 16))(img)
        return img, self.image_names[idx]

def get_inference_data(infer_path=args.infer_path):
    transform = transforms.Compose([transforms.ToTensor()])
    infer_dataset = InferenceDataset(infer_path, transform)
    infer_loader = DataLoader(dataset=infer_dataset, batch_size=1, num_workers=4)
    print('len(infer_loader):', len(infer_loader))
    logging.info('len(infer_loader): %d', len(infer_loader))
    return infer_loader

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
            sub_data, positions = split_image_overlap(inputs, crop_size=args.vit_img_size, overlap_size=args.overlap_size)
            for i, sub in enumerate(sub_data):
                out_sub = net(sub)
                out_sub = net_1(out_sub)
                sub_data[i] = out_sub
            outputs = merge_image_overlap(sub_data, positions, crop_size=args.vit_img_size, resolution=(B, C, H, W), overlap_size=args.overlap_size, blend_mode='gaussian')
            psnr_val = compute_psnr(outputs, gt)
            ssim_val = compute_ssim(outputs, gt)
            total_psnr += psnr_val
            total_ssim += ssim_val
            count += 1
            save_path = os.path.join(val_save_dir, f"{fname[0]}_iter{iteration}.png")
            torchvision.utils.save_image(outputs.cpu()[0], save_path)
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
            sub_data, positions = split_image_overlap(inputs, crop_size=args.vit_img_size, overlap_size=args.overlap_size)
            for i, sub in enumerate(sub_data):
                out_sub = net(sub)
                out_sub = net_1(out_sub)
                sub_data[i] = out_sub
            outputs = merge_image_overlap(sub_data, positions, crop_size=args.vit_img_size, resolution=(B, C, H, W), overlap_size=args.overlap_size, blend_mode='gaussian')
            save_path = os.path.join(save_dir, fname[0])
            torchvision.utils.save_image(outputs.cpu()[0], save_path)
            print(f"Saved inference output: {save_path}")

def print_param_number(net):
    total_params = sum(param.numel() for param in net.parameters())
    print('#generator parameters:', total_params)
    logging.info('#generator parameters: %d', total_params)

VAL_SAVE_DIR = os.path.join(SAVE_PATH, "validation_results")

if __name__ == '__main__':
    wandb.init(project="nafnet_wloss", name=args.experiment_name, config=vars(args))
    if args.Flag_multi_scale:
        net_1 = NAFNet(img_channel=args.img_channel, width=args.base_channel, middle_blk_num=args.num_res, enc_blk_nums=args.enc_blks, dec_blk_nums=args.dec_blks, global_residual=False)
    else:
        net_1 = NAFNet(img_channel=args.img_channel, width=args.base_channel, middle_blk_num=args.num_res, enc_blk_nums=args.enc_blks, dec_blk_nums=args.dec_blks, global_residual=False)
    net = MaskedAutoencoderViT(patch_size=args.vit_patch_size, embed_dim=args.vit_embed_dim, depth=args.vit_depth, num_heads=args.vit_num_heads, decoder_embed_dim=args.vit_decoder_embed_dim, decoder_depth=args.vit_decoder_depth, decoder_num_heads=args.vit_decoder_num_heads, mlp_ratio=args.vit_mlp_ratio, norm_layer=partial(nn.LayerNorm, eps=1e-6))
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
    scheduler = CosineAnnealingWarmRestarts(optimizer=optim.Adam(net_1.parameters(), lr=args.learning_rate, betas=(0.9, 0.999)), T_0=args.T_period, T_mult=1)
    if args.optim.lower() == 'adamw':
        optimizerG = optim.AdamW(net.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))
    elif args.optim.lower() == 'lion':
        optimizerG = Lion(net.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))
    else:
        optimizerG = optim.Adam(net_1.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))
    for param in net.parameters():
        param.requires_grad = False
    base_loss = losses.CharbonnierLoss() if args.base_loss.lower() == 'char' else (losses.WeightedCharbonnierLoss(eps=1e-4, weight=args.weight_coff) if args.base_loss.lower() == 'weightedchar' else nn.L1Loss())
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
            net_1.zero_grad()
            optimizerG.zero_grad()
            inputs = data_in.to(device)
            labels = label.to(device)
            vit_output = net(inputs)
            sub_images, positions = split_image_overlap(vit_output, crop_size=args.vit_img_size, overlap_size=args.overlap_size)
            if args.Flag_process_split_image_with_model_parallel:
                if args.Flag_multi_scale:
                    results = torch.stack([net_1(sub) for sub in sub_images])
                else:
                    results = torch.stack([net_1(sub) for sub in sub_images])
            else:
                results = torch.stack([net_1(sub) for sub in sub_images])
            train_output = merge_image_overlap(results, positions, crop_size=args.vit_img_size,
                                                resolution=(inputs.size(0), inputs.size(1), inputs.size(2), inputs.size(3)),
                                                overlap_size=args.overlap_size, blend_mode='gaussian').to(device)
            loss1 = base_loss(train_output, labels)
            fft_loss = losses.fftLoss()(train_output, labels)
            loss_total = loss1 + args.fft_loss_weight * fft_loss
            loss_addition = args.addition_loss_coff * criterion(train_output, labels) if criterion is not None else 0
            loss_total = loss_total + loss_addition
            Avg_Meters_training.update({'total_loss': loss_total.item()})
            loss_total.backward()
            optimizerG.step()
            global_iter += 1
            if (i + 1) % args.print_frequency == 0 and i > 1:
                psnr_val = compute_psnr(train_output, labels)
                ssim_val = compute_ssim(train_output, labels)
                print("Iteration:%d, [lr: %.7f], [loss_total: %.5f], PSNR: %.2f, SSIM: %.4f" %
                      (global_iter, optimizerG.param_groups[0]["lr"], loss_total.item(), psnr_val, ssim_val))
                wandb.log({
                    "iteration": global_iter,
                    "loss_total": loss_total.item(),
                    "loss_char": loss1.item(),
                    "loss_fft": fft_loss.item(),
                    "iter_psnr": psnr_val,
                    "iter_ssim": ssim_val
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
        scheduler.step()
    torch.save(net_1.state_dict(), os.path.join(SAVE_PATH, "nafnet_stage2_wloss_save_overlap.pth"))
    print("Training complete: NAFNet model (overlap) saved.")
    logging.info("Training complete: NAFNet model (overlap) saved.")
    wandb.finish()
