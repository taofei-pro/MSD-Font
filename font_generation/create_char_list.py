#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
创建完整的6763个汉字字符列表
"""

import json
import os

# 输出文件路径
output_file = "all_chars.txt"
output_json = "all_chars.json"

# 常用汉字6763个
chars = []

# GB2312一级汉字（3755个）
for i in range(0xB0A1, 0xD7F9):
    try:
        char = bytes.fromhex(f'{i:04x}').decode('gb2312')
        chars.append(char)
    except:
        pass

# 剩余汉字
for i in range(0xD8A1, 0xF7FE):
    try:
        char = bytes.fromhex(f'{i:04x}').decode('gb2312')
        chars.append(char)
    except:
        pass

# 保存到文本文件
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(''.join(chars))

# 保存到JSON文件（MSD-Font可能需要）
with open(output_json, 'w', encoding='utf-8') as f:
    json.dump(chars, f, ensure_ascii=False)

print(f'已生成 {len(chars)} 个汉字')
print(f'文本文件保存为: {output_file}')
print(f'JSON文件保存为: {output_json}')
