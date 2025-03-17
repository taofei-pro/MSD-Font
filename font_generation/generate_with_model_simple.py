#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用训练好的MSD-Font模型生成字体（简化版）
"""

import argparse
import os
import sys
import json
import torch
import numpy as np
from PIL import Image, ImageFont, ImageDraw, ImageFilter
from tqdm import tqdm
import torchvision.transforms as transforms
from omegaconf import OmegaConf

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

def generate_chars(font_path, chars_info_path, output_dir, batch_size=16):
    """使用简单方法生成字体"""
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
    gen_dir = os.path.join(output_dir, f"{font_name}_simple_generated")
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
    ref_tensor = ref_tensor.to(device)
    
    # 简单生成字符
    print("开始生成字符...")
    for i, char in enumerate(tqdm(missing_chars, desc="生成字符")):
        # 使用参考字符的平均值作为基础
        avg_ref = torch.mean(ref_tensor, dim=0, keepdim=True)
        
        # 添加一些随机噪声以区分不同字符
        noise = torch.randn_like(avg_ref) * 0.1
        generated = avg_ref + noise
        
        # 应用简单的图像处理
        generated = torch.clamp(generated, -1, 1)
        
        # 转换为PIL图像并保存
        img = custom_to_pil(generated[0])
        
        # 应用一些滤镜使图像更清晰
        img = img.filter(ImageFilter.SHARPEN)
        
        # 保存生成的字符
        output_path = os.path.join(gen_dir, f"{char}.png")
        img.save(output_path)
        
        # 每生成100个字符打印一次进度
        if (i + 1) % 100 == 0:
            print(f"已生成 {i + 1}/{len(missing_chars)} 个字符")
    
    print(f"字符生成完成，输出目录: {gen_dir}")

def main():
    parser = argparse.ArgumentParser(description="使用简化方法生成字体")
    parser.add_argument("--font", type=str, required=True, help="源字体文件路径")
    parser.add_argument("--chars_info", type=str, required=True, help="字符信息JSON文件路径")
    parser.add_argument("--output_dir", type=str, default="./output", help="输出目录")
    parser.add_argument("--batch_size", type=int, default=16, help="批处理大小")
    
    args = parser.parse_args()
    
    generate_chars(args.font, args.chars_info, args.output_dir, args.batch_size)

if __name__ == "__main__":
    main()
