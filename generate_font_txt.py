#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
from pathlib import Path
from fontTools.ttLib import TTFont

def get_chars_from_font(font_path):
    """从TTF字体文件中提取所有可用字符"""
    try:
        font = TTFont(font_path)
        chars = []
        
        # 检查cmap表
        for table in font['cmap'].tables:
            if table.isUnicode():
                for code, _ in table.cmap.items():
                    # 只保留汉字和基本符号 (0x4E00-0x9FFF是中文汉字范围)
                    if 0x4E00 <= code <= 0x9FFF or code < 128:
                        chars.append(chr(code))
        
        return chars
    except Exception as e:
        print(f"处理字体文件 {font_path} 时出错: {e}")
        return []

def main():
    # 加载训练字符集
    train_chars_path = "/home/zihun/workspace/MSD-Font/fontdata_example/chn/train_chars.json"
    with open(train_chars_path, 'r', encoding='utf-8') as f:
        train_chars = json.load(f)
    
    # 字体目录
    font_dir = "/home/zihun/workspace/MSD-Font/fontdata_example/chn/ttfs/train_font"
    
    # 获取所有TTF文件
    ttf_files = [f for f in os.listdir(font_dir) if f.endswith('.ttf')]
    
    for ttf_file in ttf_files:
        ttf_path = os.path.join(font_dir, ttf_file)
        txt_path = os.path.join(font_dir, ttf_file.replace('.ttf', '.txt'))
        
        # 如果txt文件已存在，跳过
        if os.path.exists(txt_path):
            print(f"{txt_path} 已存在，跳过")
            continue
        
        # 从字体中提取字符
        font_chars = get_chars_from_font(ttf_path)
        
        # 与训练字符集取交集
        supported_chars = set(font_chars).intersection(set(train_chars))
        
        # 如果支持的字符太少，使用所有训练字符
        if len(supported_chars) < 10:
            print(f"警告: {ttf_file} 支持的训练字符太少 ({len(supported_chars)}), 使用所有训练字符")
            supported_chars = train_chars
        
        # 写入txt文件
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(''.join(supported_chars))
        
        print(f"已创建 {txt_path}, 包含 {len(supported_chars)} 个字符")

if __name__ == "__main__":
    main()
