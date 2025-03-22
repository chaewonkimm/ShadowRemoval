# MatteViT: Shadow-Aware Transformer for High-Fidelity Shadow Removal

We propose a novel shadow removal framework, **MatteViT**, designed for the **New Trends in Image Restoration and Enhancement workshop and associated challenges in conjunction with CVPR 2025**. <br> 
MatteViT integrates shadow matte guidance with advanced neural architectures to effectively address the challenging task of shadow removal.

![Figure](images/Fig1.png)

## Requirements
To get started, install all requirements using:
```
pip install -r requirements.txt
```
<br>

## Pretrained Models
Download the pre-trained models from the following Google Drive links:
- [Shadow Matte Generator](https://drive.google.com/file/d/1x2VQQX3KQlGmoONdZ-sRBvoIgb5gW9dP/view?usp=sharing)
- [MatteViT](https://drive.google.com/file/d/1_xpq4dE1GHmo6lHfzDUQk5e6GeI2DuAs/view?usp=sharing)
- [Spatial NAFNet](https://drive.google.com/file/d/1mWsq7EVt79gjTF0S61iTScGJyqyUbL4I/view?usp=sharing)

<br>

## Usage
You can run inference using the pre-trained model with the following command:
```
python inference.py --vit_checkpoint [PATH_TO_VIT_CHECKPOINT] \
                    --nafnet_checkpoint [PATH_TO_NAFNET_CHECKPOINT] \
                    --matte_generator_checkpoint [PATH_TO_MATTE_GENERATOR_CHECKPOINT] \
                    --input_dir [INPUT_DIRECTORY] \
                    --output_dir [OUTPUT_DIRECTORY]
```
<br>

## BibTeX
```
@InProceedings{Lu_2024_CVPR,
    author    = {Lu, Xin and Zhu, Yurui and Wang, Xi and Li, Dong and Xiao, Jie and Zhang, Yunpeng and Fu, Xueyang and Zha, Zheng-Jun},
    title     = {HirFormer: Dynamic High Resolution Transformer for Large-Scale Image Shadow Removal},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2024},
    pages     = {6513-6523}
}
```
