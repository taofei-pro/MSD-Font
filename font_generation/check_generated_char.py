#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
检查生成的字符
"""

import argparse
import os
import sys
from PIL import Image, ImageFont, ImageDraw
import matplotlib.pyplot as plt

def render_char(font_path, char, size=(128, 128), pad=20):
    """渲染字体中的字符"""
    font_size = 150
    font = ImageFont.truetype(font_path, size=font_size)
    
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

def compare_chars(font_path, char, generated_dir):
    """比较源字体和生成的字符"""
    # 获取字符的Unicode码点
    unicode_hex = f"{ord(char):x}"
    generated_path = os.path.join(generated_dir, f"{unicode_hex}.png")
    
    # 检查生成的字符是否存在
    if not os.path.exists(generated_path):
        print(f"错误: 生成的字符 {char} (U+{unicode_hex.upper()}) 不存在")
        return
    
    # 渲染源字体中的字符
    try:
        source_img = render_char(font_path, char)
    except Exception as e:
        print(f"错误: 无法渲染源字体中的字符 {char}: {e}")
        source_img = Image.new("L", (128, 128), 255)
    
    # 加载生成的字符
    try:
        generated_img = Image.open(generated_path)
    except Exception as e:
        print(f"错误: 无法加载生成的字符 {char}: {e}")
        return
    
    # 显示比较结果
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(source_img, cmap='gray')
    plt.title(f"源字体: {char} (U+{unicode_hex.upper()})")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(generated_img, cmap='gray')
    plt.title(f"生成的字符: {char} (U+{unicode_hex.upper()})")
    plt.axis('off')
    
    plt.tight_layout()
    
    # 保存比较结果
    output_dir = os.path.dirname(generated_dir)
    comparison_path = os.path.join(output_dir, f"comparison_{unicode_hex}.png")
    plt.savefig(comparison_path)
    print(f"比较结果已保存到: {comparison_path}")
    
    # 关闭图形
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="检查生成的字符")
    parser.add_argument("--font", type=str, required=True, help="源字体文件路径")
    parser.add_argument("--char", type=str, required=True, help="要检查的字符")
    parser.add_argument("--generated_dir", type=str, required=True, help="生成的字符目录")
    
    args = parser.parse_args()
    
    compare_chars(args.font, args.char, args.generated_dir)

if __name__ == "__main__":
    main()
