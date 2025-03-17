#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
简化版的MSD-Font字体生成脚本
直接使用训练好的模型，不依赖于额外的检查点文件
"""

import argparse
import os
import sys
import json
import torch
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from tqdm import tqdm
from pathlib import Path
import shutil
import torchvision.transforms as transforms

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入MSD-Font相关模块
from ldm.models.diffusion.MSDFont_ddpm import MSDFont_train_stage1_rec_model
from ldm.models.diffusion.MSDFont_ddpm import MSDFont_train_stage1_trans_model_Gencase as MSDFont_train_stage1_trans_model
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

def load_models(rec_model_path, trans_model_path):
    """加载模型"""
    print("正在加载模型...")
    
    # 创建样式阶段配置
    style_stage_config = {
        "target": "ldm.modules.encoders.modules.FrozenCLIPImageEmbedder",
        "params": {
            "model": "ViT-L/14",
            "jit": False,
            "device": "cuda"
        }
    }
    
    # 创建转换模型配置
    trans_model_config = {
        "target": "ldm.models.diffusion.MSDFont_ddpm.MSDFont_train_stage1_trans_model_Gencase",
        "params": {
            "parameterization": "x0",
            "linear_start": 0.00085,
            "linear_end": 0.0120,
            "timesteps": 1000
        }
    }
    
    # 初始化第二阶段重建模型
    rec_model = MSDFont_train_stage2_rec_model_distri(
        style_stage_config=style_stage_config,
        trans_model_config=trans_model_config,
        parameterization="x0",
        linear_start=0.00085,
        linear_end=0.0120,
        num_timesteps_cond=1,
        log_every_t=200,
        timesteps=1000,
        edit_t1=200,
        edit_t2=800,
        first_stage_key="image",
        cond_stage_key="char_img",
        image_size=16,
        channels=4,
        cond_stage_trainable=False,
        conditioning_key="crossattn",
        monitor="val/loss_simple_ema",
        scale_factor=0.18215,
        use_ema=False,
        device2="cuda:0"
    )
    
    # 加载第二阶段模型权重
    rec_model.load_state_dict(torch.load(rec_model_path, weights_only=False)["state_dict"], strict=False)
    rec_model.eval()
    rec_model.cuda()
    
    # 初始化第一阶段转换模型
    trans_model = MSDFont_train_stage1_trans_model(
        parameterization="x0",
        linear_start=0.00085,
        linear_end=0.0120,
        num_timesteps_cond=1,
        log_every_t=200,
        timesteps=1000,
        edit_t1=200,
        edit_t2=800,
        first_stage_key="image",
        cond_stage_key="char_img",
        image_size=16,
        channels=4,
        cond_stage_trainable=False,
        conditioning_key="crossattn",
        monitor="val/loss_simple_ema",
        scale_factor=0.18215,
        use_ema=False
    )
    
    # 加载第一阶段模型权重
    sd = torch.load(trans_model_path, weights_only=False)
    if "state_dict" in list(sd.keys()):
        sd = sd["state_dict"]
    trans_model.load_state_dict(sd, strict=False)
    trans_model.eval()
    trans_model.cuda()
    
    return rec_model, trans_model

def generate_font(source_font_path, target_chars, output_dir, rec_model, trans_model, batch_size=8):
    """生成字体"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载源字体
    source_font = read_font(source_font_path)
    font_name = os.path.basename(source_font_path).split('.')[0]
    
    # 创建输出目录
    gen_dir = os.path.join(output_dir, f"{font_name}_generated")
    os.makedirs(gen_dir, exist_ok=True)
    
    # 获取源字体中已有的字符
    source_chars = []
    for char in tqdm(target_chars, desc="分析源字体"):
        try:
            img = render(source_font, char)
            # 检查是否是空白图像（全白）
            if np.mean(np.array(img)) < 250:  # 如果平均像素值小于250，认为不是空白
                source_chars.append(char)
        except:
            continue
    
    print(f"源字体包含 {len(source_chars)} 个有效字符")
    
    if len(source_chars) < 10:
        print("警告：源字体中有效字符太少，可能无法生成高质量结果")
        return
    
    # 准备要生成的字符（排除源字体已有的字符）
    chars_to_generate = [c for c in target_chars if c not in source_chars]
    print(f"需要生成 {len(chars_to_generate)} 个字符")
    
    # 批量生成
    for i in tqdm(range(0, len(chars_to_generate), batch_size), desc="生成字体"):
        batch_chars = chars_to_generate[i:i+batch_size]
        batch_size_actual = len(batch_chars)
        
        # 随机选择源字体中的参考字符
        ref_chars = np.random.choice(source_chars, min(batch_size_actual, len(source_chars)), replace=True)
        
        # 准备源字体图像
        ref_images = []
        for char in ref_chars:
            img = render(source_font, char)
            ref_images.append(transform(img).unsqueeze(0))
        
        if not ref_images:
            continue
            
        ref_tensor = torch.cat(ref_images, dim=0).cuda()
        
        # 编码参考图像
        with torch.no_grad():
            ref_posterior = rec_model.encode_first_stage(ref_tensor)
            ref_latent = rec_model.get_first_stage_encoding(ref_posterior).detach()
            
            style1, _ = rec_model.style_stage_model.encode(ref_tensor)
            style2, _ = trans_model.style_stage_model.encode(ref_tensor)
            
            # 生成过程
            nbs, nc, nh, nw = ref_latent.shape
            shape = [nbs, 4, 16, 16]
            
            c1 = dict(zf=[ref_latent], zcs=[style1], zcf=[style1])
            c2 = dict(zf=[ref_latent], zcs=[style2], zcf=[style2])
            
            z0, _ = trans_model.progressive_denoising(rec_model, c1, c2, shape=shape)
            gen_images = rec_model.decode_first_stage(z0)
            
            # 保存生成的图像
            for j, gen_img in enumerate(gen_images):
                if j >= len(batch_chars):
                    break
                    
                char = batch_chars[j]
                try:
                    # 保存为PNG
                    save_path = os.path.join(gen_dir, f"{ord(char):x}.png")
                    img = custom_to_pil(gen_img)
                    img.save(save_path)
                except Exception as e:
                    print(f"保存字符 {char} 时出错: {e}")
    
    print(f"字体生成完成！输出目录: {gen_dir}")
    return gen_dir

def main():
    parser = argparse.ArgumentParser(description="MSD-Font 简化版字体生成工具")
    parser.add_argument("--source_font", type=str, required=True, help="源字体文件路径")
    parser.add_argument("--output_dir", type=str, default="./output", help="输出目录")
    parser.add_argument("--char_list", type=str, help="字符列表文件路径（文本文件或JSON）")
    parser.add_argument("--batch_size", type=int, default=8, help="批处理大小")
    parser.add_argument("--rec_model", type=str, default="/home/zihun/workspace/MSD-Font/logs/2025-03-13T12-17-14_MSDFont_Train_Stage2_rec_model_predx0_miniUnet_distri_updated/checkpoints/last.ckpt", help="重建模型路径")
    parser.add_argument("--trans_model", type=str, default="/home/zihun/workspace/MSD-Font/logs/2025-03-12T18-11-23_MSDFont_Train_Stage1_trans_model_predx0_miniUnet/checkpoints/last.ckpt", help="转换模型路径")
    
    args = parser.parse_args()
    
    # 加载字符列表
    if not args.char_list:
        # 创建默认字符列表
        chars = []
        # GB2312一级汉字（3755个）
        for i in range(0xB0A1, 0xD7F9):
            try:
                chars.append(bytes.fromhex(f'{i:04x}').decode('gb2312'))
            except:
                pass
        # 剩余汉字
        for i in range(0xD8A1, 0xF7FE):
            try:
                chars.append(bytes.fromhex(f'{i:04x}').decode('gb2312'))
            except:
                pass
        print(f"使用默认字符列表，共 {len(chars)} 个字符")
    else:
        # 从文件加载字符列表
        file_ext = os.path.splitext(args.char_list)[1].lower()
        if file_ext == '.json':
            with open(args.char_list, 'r', encoding='utf-8') as f:
                chars = json.load(f)
        else:
            with open(args.char_list, 'r', encoding='utf-8') as f:
                chars = f.read()
        print(f"从 {args.char_list} 加载了 {len(chars)} 个字符")
    
    # 加载模型
    rec_model, trans_model = load_models(args.rec_model, args.trans_model)
    
    # 生成字体
    generate_font(
        args.source_font,
        chars,
        args.output_dir,
        rec_model,
        trans_model,
        args.batch_size
    )

if __name__ == "__main__":
    main()
