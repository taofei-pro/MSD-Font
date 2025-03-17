#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用训练好的MSD-Font模型生成字体 - 使用sample方法版本
"""

import argparse
import os
import sys
import json
import torch
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from tqdm import tqdm
import torchvision.transforms as transforms
from omegaconf import OmegaConf

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入模型类
from ldm.models.diffusion.MSDFont_ddpm import MSDFont_train_stage1_trans_model_Gencase
from ldm.models.diffusion.MSDFont_ddpm_distri import MSDFont_train_stage2_rec_model_distri

# 设置转换函数
class Compose(object):
    def __init__(self, tf):
        self.tf = tf

    def __call__(self, img):
        for t in self.tf:
            img = t(img)
        return img

mean = [0.5]
std = [0.5]

transform = Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

def read_font(fontfile, size=150):
    """读取字体文件"""
    font = ImageFont.truetype(str(fontfile), size=size)
    return font

def render(font, char, size=(128, 128), pad=20):
    """渲染单个字符为图像"""
    # 获取字符尺寸
    try:
        width, height = font.getsize(char)
    except:
        # 对于较新版本的PIL
        try:
            left, top, right, bottom = font.getbbox(char)
            width, height = right - left, bottom - top
        except:
            # 如果无法获取尺寸，使用默认值
            width, height = 100, 100
    
    max_size = max(width, height)

    if width < height:
        start_w = (height - width) // 2 + pad
        start_h = pad
    else:
        start_w = pad
        start_h = (width - height) // 2 + pad

    img = Image.new("L", (max_size+(pad*2), max_size+(pad*2)), 255)
    draw = ImageDraw.Draw(img)
    draw.text((start_w, start_h), char, font=font, fill=0)
    img = img.resize(size, Image.LANCZOS)
    
    return img

def custom_to_pil(x):
    """将张量转换为PIL图像"""
    x = x.detach().cpu()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.) / 2.
    x = x.permute(1, 2, 0).numpy()
    x = (255 * x).astype(np.uint8)
    if x.shape[2] == 1:
        x = x[:, :, 0]
    return Image.fromarray(x)

def load_model_from_config(config, ckpt, verbose=False):
    """从配置和检查点加载模型"""
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.eval()
    return model

def instantiate_from_config(config):
    """从配置实例化模型"""
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

def get_obj_from_str(string, reload=False):
    """从字符串获取对象"""
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def generate_with_model_sample(font_path, chars_info_path, output_dir, rec_model_path, trans_model_path, batch_size=16):
    """使用训练好的模型生成字体 - 使用sample方法版本"""
    # 加载字符信息
    with open(chars_info_path, 'r', encoding='utf-8') as f:
        chars_info = json.load(f)
    
    valid_chars = chars_info['valid_chars']
    missing_chars = chars_info['missing_chars']
    
    print(f"源字体有效字符: {len(valid_chars)}")
    print(f"需要生成的字符: {len(missing_chars)}")
    
    if len(valid_chars) < 10:
        print("警告: 源字体中有效字符太少，可能无法生成高质量结果")
        return
    
    # 加载源字体
    source_font = read_font(font_path)
    font_name = os.path.basename(font_path).split('.')[0]
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    gen_dir = os.path.join(output_dir, f"{font_name}_model_generated")
    os.makedirs(gen_dir, exist_ok=True)
    
    # 准备源字体图像
    ref_images = []
    ref_chars = []
    
    print("准备参考字符...")
    for char in tqdm(valid_chars[:100], desc="准备参考字符"):  # 使用前100个有效字符作为参考
        try:
            img = render(source_font, char)
            ref_images.append(transform(img).unsqueeze(0))
            ref_chars.append(char)
        except Exception as e:
            print(f"处理参考字符 {char} 时出错: {e}")
    
    if not ref_images:
        print("错误: 无法准备参考字符图像")
        return
    
    ref_tensor = torch.cat(ref_images, dim=0)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    try:
        print("加载模型...")
        
        # 加载第一阶段转换模型配置
        print("加载第一阶段转换模型配置...")
        config_path = "../configs/MSDFont/MSDFont_Train_Stage1_trans_model_predx0_miniUnet.yaml"
        config = OmegaConf.load(config_path)
        
        # 加载第一阶段转换模型
        print("加载第一阶段转换模型...")
        trans_model = MSDFont_train_stage1_trans_model_Gencase(
            style_stage_config=OmegaConf.create({
                "target": "ldm.models.MSDFont_encoder.miniUNet_enc_128"
            }),
            first_stage_config=OmegaConf.create({
                "target": "ldm.models.autoencoder.AutoencoderKL",
                "params": {
                    "embed_dim": 4,
                    "monitor": "val/rec_loss",
                    "ddconfig": {
                        "double_z": True,
                        "z_channels": 4,
                        "resolution": 256,
                        "in_channels": 3,
                        "out_ch": 3,
                        "ch": 128,
                        "ch_mult": [1, 2, 4, 4],
                        "num_res_blocks": 2,
                        "attn_resolutions": [],
                        "dropout": 0.0
                    },
                    "lossconfig": {
                        "target": "torch.nn.Identity"
                    }
                }
            }),
            cond_stage_config="__is_first_stage__",
            unet_config=OmegaConf.create({
                "target": "ldm.modules.diffusionmodules.openaimodel.UNetModel",
                "params": {
                    "use_checkpoint": True,
                    "use_fp16": False,
                    "image_size": 16,
                    "in_channels": 4,
                    "out_channels": 4,
                    "model_channels": 128,
                    "attention_resolutions": [2, 1, 1],
                    "num_res_blocks": 2,
                    "channel_mult": [1, 1, 2, 2],
                    "num_head_channels": 64,
                    "use_spatial_transformer": True,
                    "use_linear_in_transformer": True,
                    "transformer_depth": 1,
                    "context_dim": 128,
                    "legacy": False
                }
            }),
            timesteps=1000,
            beta_schedule="linear",
            linear_start=0.00085,
            linear_end=0.0120,
            conditioning_key="crossattn",
            parameterization="x0"
        )
        
        # 加载第二阶段重建模型
        print("加载第二阶段重建模型...")
        rec_model = MSDFont_train_stage2_rec_model_distri(
            style_stage_config=OmegaConf.create({
                "target": "ldm.models.MSDFont_encoder.miniUNet_enc_128"
            }),
            first_stage_config=OmegaConf.create({
                "target": "ldm.models.autoencoder.AutoencoderKL",
                "params": {
                    "embed_dim": 4,
                    "monitor": "val/rec_loss",
                    "ddconfig": {
                        "double_z": True,
                        "z_channels": 4,
                        "resolution": 256,
                        "in_channels": 3,
                        "out_ch": 3,
                        "ch": 128,
                        "ch_mult": [1, 2, 4, 4],
                        "num_res_blocks": 2,
                        "attn_resolutions": [],
                        "dropout": 0.0
                    },
                    "lossconfig": {
                        "target": "torch.nn.Identity"
                    }
                }
            }),
            cond_stage_config="__is_first_stage__",
            unet_config=OmegaConf.create({
                "target": "ldm.modules.diffusionmodules.openaimodel.UNetModel",
                "params": {
                    "use_checkpoint": True,
                    "use_fp16": False,
                    "image_size": 16,
                    "in_channels": 4,
                    "out_channels": 4,
                    "model_channels": 128,
                    "attention_resolutions": [2, 1, 1],
                    "num_res_blocks": 2,
                    "channel_mult": [1, 1, 2, 2],
                    "num_head_channels": 64,
                    "use_spatial_transformer": True,
                    "use_linear_in_transformer": True,
                    "transformer_depth": 1,
                    "context_dim": 128,
                    "legacy": False
                }
            }),
            trans_model_config=OmegaConf.create({
                "config_path": "../configs/MSDFont/MSDFont_Train_Stage1_trans_model_predx0_miniUnet.yaml",
                "model_path": trans_model_path
            }),
            timesteps=1000,
            beta_schedule="linear",
            linear_start=0.00085,
            linear_end=0.0120,
            conditioning_key="crossattn",
            parameterization="x0"
        )
        
        # 加载模型权重
        print("加载模型权重...")
        trans_checkpoint = torch.load(trans_model_path, map_location=device)
        rec_checkpoint = torch.load(rec_model_path, map_location=device)
        
        # 加载权重
        if 'state_dict' in trans_checkpoint:
            trans_model.load_state_dict(trans_checkpoint['state_dict'], strict=False)
        else:
            trans_model.load_state_dict(trans_checkpoint, strict=False)
            
        if 'state_dict' in rec_checkpoint:
            rec_model.load_state_dict(rec_checkpoint['state_dict'], strict=False)
        else:
            rec_model.load_state_dict(rec_checkpoint, strict=False)
        
        # 将模型移动到设备上
        rec_model = rec_model.to(device)
        trans_model = trans_model.to(device)
        
        # 设置为评估模式
        rec_model.eval()
        trans_model.eval()
        
        print("模型加载成功")
        
    except Exception as e:
        print(f"加载模型时出错: {e}")
        import traceback
        traceback.print_exc()
        return
    
    if torch.cuda.is_available():
        ref_tensor = ref_tensor.to(device)
        print("使用CUDA加速")
    
    # 批量生成缺失字符
    print("开始生成字符...")
    for i in tqdm(range(0, len(missing_chars), batch_size), desc="生成字符"):
        batch_chars = missing_chars[i:i+batch_size]
        
        for j, char in enumerate(batch_chars):
            try:
                # 随机选择参考字符
                ref_idx = np.random.randint(0, len(ref_tensor))
                ref_img = ref_tensor[ref_idx:ref_idx+1]
                ref_char = ref_chars[ref_idx]
                
                # 使用模型生成新字符
                with torch.no_grad():
                    # 使用模型的sample方法
                    shape = [1, 4, 32, 32]
                    samples = rec_model.sample(
                        batch_size=shape[0],
                        x_shape=shape[1:],
                        conditioning=ref_img,
                        unconditional_guidance_scale=1.0,
                        unconditional_conditioning=None,
                        eta=0.0
                    )
                    
                    # 使用转换模型
                    generated_img = trans_model.sample(
                        batch_size=shape[0],
                        x_shape=shape[1:],
                        conditioning=ref_img,
                        unconditional_guidance_scale=1.0,
                        unconditional_conditioning=None,
                        eta=0.0
                    )
                
                # 保存生成的图像
                save_path = os.path.join(gen_dir, f"{ord(char):x}.png")
                img = custom_to_pil(generated_img[0])
                img.save(save_path)
                
                # 每生成100个字符打印一次进度
                if (j + 1) % 100 == 0:
                    print(f"已生成 {i + j + 1}/{len(missing_chars)} 个字符")
                
            except Exception as e:
                print(f"生成字符 {char} 时出错: {e}")
                import traceback
                traceback.print_exc()
    
    # 复制源字体中已有的字符
    for char in tqdm(valid_chars, desc="复制已有字符"):
        try:
            img = render(source_font, char)
            save_path = os.path.join(gen_dir, f"{ord(char):x}.png")
            img.save(save_path)
        except Exception as e:
            print(f"复制字符 {char} 时出错: {e}")
    
    print(f"字体生成完成！输出目录: {gen_dir}")
    print(f"共生成 {len(missing_chars)} 个新字符，复制 {len(valid_chars)} 个已有字符")
    return gen_dir

def main():
    parser = argparse.ArgumentParser(description="使用训练好的模型生成字体 - 使用sample方法版本")
    parser.add_argument("--font", type=str, required=True, help="源字体文件路径")
    parser.add_argument("--chars_info", type=str, required=True, help="字符信息JSON文件路径")
    parser.add_argument("--output_dir", type=str, default="./output", help="输出目录")
    parser.add_argument("--rec_model", type=str, 
                        default="/home/zihun/workspace/MSD-Font/logs/2025-03-13T12-17-14_MSDFont_Train_Stage2_rec_model_predx0_miniUnet_distri_updated/checkpoints/last.ckpt", 
                        help="重建模型路径")
    parser.add_argument("--trans_model", type=str, 
                        default="/home/zihun/workspace/MSD-Font/logs/2025-03-12T18-11-23_MSDFont_Train_Stage1_trans_model_predx0_miniUnet/checkpoints/last.ckpt", 
                        help="转换模型路径")
    parser.add_argument("--batch_size", type=int, default=16, help="批处理大小")
    
    args = parser.parse_args()
    
    generate_with_model_sample(
        args.font,
        args.chars_info,
        args.output_dir,
        args.rec_model,
        args.trans_model,
        args.batch_size
    )

if __name__ == "__main__":
    main()
