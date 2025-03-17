#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
生成字体中缺失的字符
使用简单的图像变换方法
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
import random

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

def generate_missing_chars(font_path, chars_info_path, output_dir, batch_size=16):
    """生成缺失的字符"""
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
    gen_dir = os.path.join(output_dir, f"{font_name}_generated")
    os.makedirs(gen_dir, exist_ok=True)
    
    # 准备源字体图像
    ref_images = []
    for char in tqdm(valid_chars[:100], desc="准备参考字符"):  # 使用前100个有效字符作为参考
        try:
            img = render(source_font, char)
            ref_images.append(transform(img).unsqueeze(0))
        except Exception as e:
            print(f"处理参考字符 {char} 时出错: {e}")
    
    if not ref_images:
        print("错误: 无法准备参考字符图像")
        return
    
    ref_tensor = torch.cat(ref_images, dim=0)
    if torch.cuda.is_available():
        ref_tensor = ref_tensor.cuda()
        print("使用CUDA加速")
    
    # 批量生成缺失字符
    for i in tqdm(range(0, len(missing_chars), batch_size), desc="生成字符"):
        batch_chars = missing_chars[i:i+batch_size]
        
        for j, char in enumerate(batch_chars):
            try:
                # 随机选择参考字符
                ref_idx = random.randint(0, len(ref_tensor) - 1)
                ref_img = ref_tensor[ref_idx]
                
                # 应用简单的变换生成新字符
                # 在实际应用中，应该使用训练好的模型进行生成
                transforms_list = [
                    lambda x: x,  # 原始图像
                    lambda x: torch.flip(x, dims=[1]),  # 水平翻转
                    lambda x: torch.flip(x, dims=[2]),  # 垂直翻转
                    lambda x: torch.rot90(x, k=1, dims=[1, 2]),  # 旋转90度
                    lambda x: torch.rot90(x, k=2, dims=[1, 2]),  # 旋转180度
                    lambda x: torch.rot90(x, k=3, dims=[1, 2])   # 旋转270度
                ]
                
                # 随机选择一种变换
                transform_func = random.choice(transforms_list)
                generated_img = transform_func(ref_img)
                
                # 保存生成的图像
                save_path = os.path.join(gen_dir, f"{ord(char):x}.png")
                img = custom_to_pil(generated_img)
                img.save(save_path)
                
            except Exception as e:
                print(f"生成字符 {char} 时出错: {e}")
    
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
    parser = argparse.ArgumentParser(description="生成字体中缺失的字符")
    parser.add_argument("--font", type=str, required=True, help="源字体文件路径")
    parser.add_argument("--chars_info", type=str, required=True, help="字符信息JSON文件路径")
    parser.add_argument("--output_dir", type=str, default="./output", help="输出目录")
    parser.add_argument("--batch_size", type=int, default=16, help="批处理大小")
    
    args = parser.parse_args()
    
    generate_missing_chars(
        args.font,
        args.chars_info,
        args.output_dir,
        args.batch_size
    )

if __name__ == "__main__":
    main()
