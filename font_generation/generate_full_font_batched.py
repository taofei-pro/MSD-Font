#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MSD-Font 批量字体生成脚本
用于从有限字符集的字体生成完整字库
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
from omegaconf import OmegaConf
import torchvision.transforms as transforms

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入MSD-Font相关模块
from main import instantiate_from_config


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

transform = Compose(
    [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
)


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

    img = Image.new("L", (max_size + (pad * 2), max_size + (pad * 2)), 255)
    draw = ImageDraw.Draw(img)
    draw.text((start_w, start_h), char, font=font, fill=0)
    img = img.resize(size, Image.LANCZOS)

    return img


def custom_to_pil(x):
    """将张量转换为PIL图像"""
    x = x.detach().cpu()
    x = torch.clamp(x, -1.0, 1.0)
    x = (x + 1.0) / 2.0
    x = x.permute(1, 2, 0).numpy()
    x = (255 * x).astype(np.uint8)
    if x.shape[2] == 1:
        x = x[:, :, 0]
    return Image.fromarray(x)


def load_models(config_rec, rec_model_path, config_trans, trans_model_path):
    """加载模型"""
    print("正在加载模型...")

    # 加载配置
    config_rec = OmegaConf.load(config_rec)
    config_trans = OmegaConf.load(config_trans)

    # 初始化模型
    rec_model = instantiate_from_config(config_rec.model)
    rec_model.load_state_dict(
        torch.load(rec_model_path, weights_only=False)["state_dict"], strict=False
    )
    rec_model.eval()
    rec_model.cuda()

    trans_model = instantiate_from_config(config_trans.model)
    sd = torch.load(trans_model_path, weights_only=False)
    if "state_dict" in list(sd.keys()):
        sd = sd["state_dict"]
    trans_model.load_state_dict(sd, strict=False)
    trans_model.eval()
    trans_model.cuda()

    return rec_model, trans_model


def generate_font(
    source_font_path, target_chars, output_dir, rec_model, trans_model, batch_size=16
):
    """生成字体"""
    os.makedirs(output_dir, exist_ok=True)

    # 加载源字体
    source_font = read_font(source_font_path)
    font_name = os.path.basename(source_font_path).split(".")[0]

    # 创建输出目录
    gen_dir = os.path.join(output_dir, f"{font_name}_generated")
    os.makedirs(gen_dir, exist_ok=True)

    # 获取源字体中已有的字符
    source_chars = []
    for char in target_chars:
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
    for i in tqdm(range(0, len(chars_to_generate), batch_size)):
        batch_chars = chars_to_generate[i : i + batch_size]

        # 随机选择源字体中的参考字符
        ref_chars = np.random.choice(
            source_chars, min(len(batch_chars), len(source_chars)), replace=True
        )

        # 准备源字体图像
        ref_images = []
        for char in ref_chars:
            img = render(source_font, char)
            ref_images.append(transform(img).unsqueeze(0))

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
    parser = argparse.ArgumentParser(description="MSD-Font 批量字体生成工具")
    parser.add_argument("--source_font", type=str, required=True, help="源字体文件路径")
    parser.add_argument("--output_dir", type=str, default="./output", help="输出目录")
    parser.add_argument(
        "--char_list", type=str, help="字符列表文件路径（文本文件或JSON）"
    )
    parser.add_argument("--batch_size", type=int, default=16, help="批处理大小")
    parser.add_argument(
        "--config_rec",
        type=str,
        default="/home/zihun/workspace/MSD-Font/configs/MSDFont/MSDFont_Eval_rec_model_predx0_miniUnet.yaml",
        help="重建模型配置文件",
    )
    parser.add_argument(
        "--rec_model",
        type=str,
        default="/home/zihun/workspace/MSD-Font/logs/2025-03-13T12-17-14_MSDFont_Train_Stage2_rec_model_predx0_miniUnet_distri/checkpoints/last.ckpt",
        help="重建模型路径",
    )
    parser.add_argument(
        "--config_trans",
        type=str,
        default="/home/zihun/workspace/MSD-Font/configs/MSDFont/MSDFont_Eval_trans_model_predx0_miniUnet.yaml",
        help="转换模型配置文件",
    )
    parser.add_argument(
        "--trans_model",
        type=str,
        default="/home/zihun/workspace/MSD-Font/logs/2025-03-12T18-11-23_MSDFont_Train_Stage1_trans_model_predx0_miniUnet/checkpoints/last.ckpt",
        help="转换模型路径",
    )

    args = parser.parse_args()

    # 加载字符列表
    if not args.char_list:
        # 创建默认字符列表
        chars = []
        # GB2312一级汉字（3755个）
        for i in range(0xB0A1, 0xD7F9):
            try:
                chars.append(bytes.fromhex(f"{i:04x}").decode("gb2312"))
            except:
                pass
        # 剩余汉字
        for i in range(0xD8A1, 0xF7FE):
            try:
                chars.append(bytes.fromhex(f"{i:04x}").decode("gb2312"))
            except:
                pass
        print(f"使用默认字符列表，共 {len(chars)} 个字符")
    else:
        # 从文件加载字符列表
        file_ext = os.path.splitext(args.char_list)[1].lower()
        if file_ext == ".json":
            with open(args.char_list, "r", encoding="utf-8") as f:
                chars = json.load(f)
        else:
            with open(args.char_list, "r", encoding="utf-8") as f:
                chars = f.read()
        print(f"从 {args.char_list} 加载了 {len(chars)} 个字符")

    # 加载模型
    rec_model, trans_model = load_models(
        args.config_rec, args.rec_model, args.config_trans, args.trans_model
    )

    # 生成字体
    generate_font(
        args.source_font,
        chars,
        args.output_dir,
        rec_model,
        trans_model,
        args.batch_size,
    )


if __name__ == "__main__":
    main()
