import time, torchvision, argparse, logging, sys, os, gc
import torch, random
import numpy as np
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from utils.UTILS1 import compute_psnr
from utils.UTILS import AverageMeters, print_args_parameters, Lion
import loss.losses as losses
from torch.utils.tensorboard import SummaryWriter
from functools import partial
from datasets.datasets_pairs import my_dataset  # 수정된 my_dataset: 하나의 root_dir를 사용
from networks.NAFNet_arch import NAFNet
from networks.MaeVit_arch import MaskedAutoencoderViT
from networks.Split_images import split_image, process_split_image_with_model, merge, process_split_image_with_model_parallel
from networks.image_utils import splitimage, mergeimage
from PIL import Image
sys.path.append(os.getcwd())

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
parser.add_argument('--vit_img_size', type=int, default=352)
parser.add_argument('--vit_grid_type', type=str, default='4x4')

parser.add_argument('--Flag_process_split_image_with_model_parallel', type=bool, default=True)
parser.add_argument('--Flag_multi_scale', type=bool, default=False)

parser.add_argument('--experiment_name', type=str, default="train_shadow_vit")
parser.add_argument('--unified_path', type=str, default='/root/autodl-tmp/SR_1/')

# 이제 GT와 입력 이미지가 같은 디렉토리에 있으므로 하나의 경로만 사용합니다.
parser.add_argument('--training_path', type=str, default='../train', help='Training images folder (contains both _in and _gt files)')

parser.add_argument('--writer_dir', type=str, default='/root/tf-logs/')
parser.add_argument('--infer_path', type=str, default='./test/input', help='Inference input images folder')

parser.add_argument('--EPOCH', type=int, default=600)
parser.add_argument('--T_period', type=int, default=50)
parser.add_argument('--BATCH_SIZE', type=int, default=24)
parser.add_argument('--overlap_size', type=int, default=0)
parser.add_argument('--Crop_patches', type=int, default=1024)
parser.add_argument('--learning_rate', type=float, default=0.0004)
parser.add_argument('--print_frequency', type=int, default=50)
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
parser.add_argument('--pre_model', type=str, default='/root/autodl-tmp/SR_1/train_shadow_vit_PL_BS.3_PS.1408_LR.2e-4_4x4_fine_tune/net_epoch_202_PSNR_23.21.pth')
parser.add_argument('--pre_model_0', type=str, default='/root/autodl-tmp/SR_1/train_shadow_vit_PL_BS.3_PS.1408_LR.2e-4_4x4_fine_tune/net_epoch_202_PSNR_23.21.pth')
parser.add_argument('--pre_model_1', type=str, default='')

parser.add_argument('--optim', type=str, default='adam')

args = parser.parse_args()
print_args_parameters(args)

if args.debug == True:
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
logging.info('======================'*2 + 'args: parameters' + '======================'*2)
for k in args.__dict__:
    logging.info(k + ": " + str(args.__dict__[k]))
logging.info('======================'*2 + 'args: parameters' + '======================'*2)

logging.info('begin training!')
print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

##############################
# Inference Dataset 정의 (라벨 없음)
##############################
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

##############################
# Inference 함수: 라벨 없는 데이터에 대해 모델 추론 후 결과 저장
##############################
def inference(net, net_1, infer_loader, save_dir):
    net.eval()
    net_1.eval()
    os.makedirs(save_dir, exist_ok=True)
    with torch.no_grad():
        for idx, (data, fname) in enumerate(infer_loader):
            inputs = Variable(data).to(device)
            B, C, H, W = inputs.shape
            sub_data, positions = splitimage(inputs, crop_size=args.vit_img_size, overlap_size=0)
            for i, sub in enumerate(sub_data):
                out_sub = net(sub)
                out_sub = net_1(out_sub)
                sub_data[i] = out_sub
            outputs = mergeimage(sub_data, positions, crop_size=args.vit_img_size, resolution=(B, C, H, W), is_mean=True)
            save_path = os.path.join(save_dir, fname[0])
            torchvision.utils.save_image(outputs.cpu()[0], save_path)
            print(f"Saved inference output: {save_path}")

##############################
# 학습 데이터 로더 (수정된 my_dataset 사용)
##############################
def get_training_data(Crop_patches=args.Crop_patches):
    # 동일 디렉토리에서 '_in' 파일만 골라내고, GT는 파일명에서 '_in'을 '_gt'로 변환합니다.
    train_dataset = my_dataset(
        root_dir=args.training_path,
        crop_size=Crop_patches,
        fix_sample_A=fix_sampleA,
        regular_aug=args.Aug_regular
    )
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.BATCH_SIZE, num_workers=8, shuffle=True)
    print('len(train_loader):', len(train_loader))
    logging.info('len(train_loader): %d', len(train_loader))
    return train_loader

def print_param_number(net):
    total_params = sum(param.numel() for param in net.parameters())
    print('#generator parameters:', total_params)
    logging.info('#generator parameters: %d', total_params)

##############################
# 메인 학습 및 Inference 실행
##############################
if __name__ == '__main__':    
    if args.Flag_multi_scale:
        net_1 = NAFNet(img_channel=6, width=args.base_channel, middle_blk_num=args.num_res,
                        enc_blk_nums=args.enc_blks, dec_blk_nums=args.dec_blks, global_residual=False)
    else:
        net_1 = NAFNet(img_channel=args.img_channel, width=args.base_channel, middle_blk_num=args.num_res,
                        enc_blk_nums=args.enc_blks, dec_blk_nums=args.dec_blks, global_residual=False)
    net = MaskedAutoencoderViT(
         patch_size=args.vit_patch_size, embed_dim=args.vit_embed_dim, depth=args.vit_depth, num_heads=args.vit_num_heads,
         decoder_embed_dim=args.vit_decoder_embed_dim, decoder_depth=args.vit_decoder_depth, decoder_num_heads=args.vit_decoder_num_heads,
         mlp_ratio=args.vit_mlp_ratio, norm_layer=partial(nn.LayerNorm, eps=1e-6))
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
    
    train_loader = get_training_data()
    if args.optim.lower() == 'adamw':
        optimizerG = optim.AdamW(net.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))
    elif args.optim.lower() == 'lion':
        optimizerG = Lion(net.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))
    else:
        optimizerG = optim.Adam(net_1.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))
    
    # ViT(net)의 파라미터 고정 (freeze)
    for param in net.parameters():
        param.requires_grad = False

    scheduler = CosineAnnealingWarmRestarts(optimizerG, T_0=args.T_period, T_mult=1)

    if args.base_loss.lower() == 'char':
        base_loss = losses.CharbonnierLoss()
    elif args.base_loss.lower() == 'weightedchar':
        base_loss = losses.WeightedCharbonnierLoss(eps=1e-4, weight=args.weight_coff)
    else:
        base_loss = nn.L1Loss()

    if args.addition_loss.lower() == 'vgg':
        criterion = losses.VGGLoss()
    elif args.addition_loss.lower() == 'ssim':
        criterion = losses.SSIMLoss()   
        
    criterion_depth = nn.L1Loss()

    running_results = {'iter_nums': 0}
    Avg_Meters_training = AverageMeters()

    for epoch in range(args.EPOCH):
        scheduler.step(epoch)
        st = time.time()
        for i, train_data in enumerate(train_loader, 0):
            data_in, label, img_name = train_data
            if i == 0:
                print(f" train_input.size: {data_in.size()}, gt.size: {label.size()}")
            running_results['iter_nums'] += 1
            net_1.train()
            net_1.zero_grad()
            optimizerG.zero_grad()
            inputs = Variable(data_in).to(device)
            labels = Variable(label).to(device)
            
            sub_images, positions = split_image(inputs, args.vit_grid_type)
            if args.Flag_process_split_image_with_model_parallel:
                if args.Flag_multi_scale:
                    results_1 = process_split_image_with_model_parallel(sub_images, net)
                    results_2 = [torch.cat((img1, img2), dim=1) for img1, img2 in zip(results_1, sub_images)]
                    results = process_split_image_with_model_parallel(results_2, net_1)
                else:
                    results = process_split_image_with_model_parallel(sub_images, net)
                    results = process_split_image_with_model_parallel(results, net_1)
            else:
                if args.Flag_multi_scale:
                    results_1 = process_split_image_with_model(sub_images, net)
                    results_2 = [torch.cat((img1, img2), dim=1) for img1, img2 in zip(results_1, sub_images)]
                    results = process_split_image_with_model(results_2, net_1)
                else:
                    results = process_split_image_with_model(sub_images, net)
                    results = process_split_image_with_model(results, net_1)
            train_output = merge(results, positions).to(device)

            loss1 = base_loss(train_output, labels)
            if args.addition_loss.lower() == 'vgg':
                loss2 = args.addition_loss_coff * criterion(train_output, labels)
                g_loss = loss1 + loss2
            elif args.addition_loss.lower() == 'ssim':
                loss2 = args.addition_loss_coff * criterion(train_output, labels)
                g_loss = loss1 + loss2
            else:
                g_loss = loss1

            Avg_Meters_training.update({'total_loss': g_loss.item()})
            g_loss.backward()
            optimizerG.step()
            if (i + 1) % args.print_frequency == 0 and i > 1:
                print("epoch:%d,[%d / %d], [lr: %.7f ],[loss: %.5f], time: %.3f" %
                      (epoch, i + 1, len(train_loader), optimizerG.param_groups[0]["lr"], g_loss.item(), time.time() - st))
                st = time.time()
                
        infer_loader = get_inference_data(infer_path=args.infer_path)
        inference_save_dir = os.path.join(SAVE_PATH, 'inference_results', f'epoch_{epoch}')
        inference(net, net_1, infer_loader, inference_save_dir)
