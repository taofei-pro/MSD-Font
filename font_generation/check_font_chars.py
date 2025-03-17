#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
检查字体文件中包含的有效字符
"""

import argparse
import os
import sys
import json
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from tqdm import tqdm
from fontTools.ttLib import TTFont

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

def check_font_chars(font_path, output_file=None):
    """检查字体文件中包含的有效字符"""
    # 使用fontTools检查字体中的字符
    ttfont = TTFont(font_path)
    cmap = ttfont.getBestCmap()
    
    # 获取字体中包含的Unicode码点
    unicode_points = set(cmap.keys())
    print(f"字体文件包含 {len(unicode_points)} 个Unicode码点")
    
    # 创建PIL字体对象进行渲染测试
    pil_font = read_font(font_path)
    
    # 创建默认字符列表（常用汉字）
    all_chars = []
    # GB2312一级汉字（3755个）
    for i in range(0xB0A1, 0xD7F9):
        try:
            all_chars.append(bytes.fromhex(f'{i:04x}').decode('gb2312'))
        except:
            pass
    # 剩余汉字
    for i in range(0xD8A1, 0xF7FE):
        try:
            all_chars.append(bytes.fromhex(f'{i:04x}').decode('gb2312'))
        except:
            pass
    print(f"待检查字符列表，共 {len(all_chars)} 个字符")
    
    # 检查字体中的有效字符
    valid_chars = []
    missing_chars = []
    
    for char in tqdm(all_chars, desc="检查字符"):
        char_code = ord(char)
        
        # 检查Unicode码点是否在字体中
        if char_code in unicode_points:
            # 进一步通过渲染测试验证
            try:
                img = render(pil_font, char)
                img_array = np.array(img)
                
                # 计算非白色像素的比例
                non_white_ratio = np.sum(img_array < 240) / img_array.size
                
                # 如果非白色像素比例大于1%，认为是有效字符
                if non_white_ratio > 0.01:
                    valid_chars.append(char)
                else:
                    missing_chars.append(char)
            except:
                missing_chars.append(char)
        else:
            missing_chars.append(char)
    
    print(f"字体中有效字符数量: {len(valid_chars)}")
    print(f"字体中缺失字符数量: {len(missing_chars)}")
    
    # 保存结果到文件
    if output_file:
        result = {
            "font_path": font_path,
            "total_unicode_points": len(unicode_points),
            "valid_chars_count": len(valid_chars),
            "missing_chars_count": len(missing_chars),
            "valid_chars": valid_chars,
            "missing_chars": missing_chars
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"结果已保存到: {output_file}")
    
    return valid_chars, missing_chars

def main():
    parser = argparse.ArgumentParser(description="检查字体文件中包含的有效字符")
    parser.add_argument("--font", type=str, required=True, help="字体文件路径")
    parser.add_argument("--output", type=str, default=None, help="输出结果文件路径")
    
    args = parser.parse_args()
    
    check_font_chars(args.font, args.output)

if __name__ == "__main__":
    main()
