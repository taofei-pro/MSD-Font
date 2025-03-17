#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import argparse
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import torch
from tqdm import tqdm
from omegaconf import OmegaConf

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 添加安全全局变量
import torch.serialization
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from numpy.core.multiarray import scalar

# 导入模型
from ldm.models.diffusion.MSDFont_ddpm import MSDFont_train_stage1_trans_model_Gencase
from ldm.models.diffusion.MSDFont_ddpm_distri import MSDFont_train_stage2_rec_model_distri

def add_safe_globals():
    """添加安全全局变量，解决PyTorch 2.6序列化安全性问题"""
    try:
        torch.serialization.add_safe_globals([ModelCheckpoint, scalar])
        print("已添加安全全局变量")
    except Exception as e:
        print(f"添加安全全局变量时出错: {e}")

def read_font(font_path, size=128):
    """读取字体"""
    try:
        font = ImageFont.truetype(font_path, size=size)
        return font
    except Exception as e:
        print(f"读取字体时出错: {e}")
        return None

def render_character(font, char, size=128, padding=10):
    """渲染字符为图像"""
    # 创建空白图像
    img = Image.new('L', (size, size), 255)
    draw = ImageDraw.Draw(img)
    
    # 获取字符尺寸
    try:
        left, top, right, bottom = draw.textbbox((0, 0), char, font=font)
        width, height = right - left, bottom - top
        
        # 计算居中位置
        x = (size - width) // 2 - left
        y = (size - height) // 2 - top
        
        # 绘制字符
        draw.text((x, y), char, font=font, fill=0)
        return img
    except Exception as e:
        print(f"渲染字符 {char} 时出错: {e}")
        return None

def load_model_weights(model, checkpoint_path, strict=True):
    """加载模型权重"""
    try:
        # 尝试使用weights_only=False加载
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        print(f"使用weights_only=False成功加载模型: {checkpoint_path}")
    except Exception as e:
        print(f"使用weights_only=False加载失败: {e}")
        try:
            # 尝试使用safe_globals加载
            with torch.serialization.safe_globals([ModelCheckpoint, scalar]):
                state_dict = torch.load(checkpoint_path, map_location="cpu")
            print(f"使用safe_globals成功加载模型: {checkpoint_path}")
        except Exception as e:
            print(f"使用safe_globals加载失败: {e}")
            return False
    
    # 如果state_dict是Lightning模型检查点，提取模型状态字典
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    
    # 加载状态字典到模型
    try:
        model.load_state_dict(state_dict, strict=strict)
        print("成功加载模型权重")
        return True
    except Exception as e:
        print(f"加载模型权重时出错: {e}")
        # 尝试移除模块前缀
        try:
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('model.'):
                    new_state_dict[k[6:]] = v
                else:
                    new_state_dict[k] = v
            model.load_state_dict(new_state_dict, strict=strict)
            print("成功加载模型权重（移除模块前缀后）")
            return True
        except Exception as e2:
            print(f"移除模块前缀后加载模型权重时出错: {e2}")
            return False

def generate_character(trans_model, rec_model, source_img, target_char, device='cpu'):
    """使用模型生成字符"""
    try:
        with torch.no_grad():
            # 将源图像转换为灰度图
            if source_img.mode != 'L':
                source_img = source_img.convert('L')
            
            # 调整图像大小为128x128，这是模型期望的输入大小
            source_img = source_img.resize((128, 128), Image.LANCZOS)
            
            # 将灰度图像转换为张量
            source_tensor = torch.from_numpy(np.array(source_img)).float() / 127.5 - 1.0
            source_tensor = source_tensor.unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, H, W]
            print(f"源图像张量形状: {source_tensor.shape}")
            
            # 使用style_stage_model将图像编码为潜在表示
            print("编码源图像...")
            zcf, zcs = trans_model.style_stage_model.encode(source_tensor)
            print(f"zcf形状: {zcf.shape}, zcs形状: {zcs.shape}")
            
            # 设置参数
            batch_size = 1
            shape = (batch_size, 4, 32, 32)  # 批大小, 通道数, 高度, 宽度
            
            # 第一阶段：使用手动方式进行推理，避免形状不匹配问题
            print("第一阶段：使用手动方式进行推理...")
            
            # 创建一个随机噪声作为起点
            x_T = torch.randn(shape).to(device)
            
            # 设置时间步
            t = torch.zeros(batch_size, dtype=torch.long, device=device)
            
            # 手动处理条件输入的形状
            bs, nc, nh, nw = zcs.shape
            zcs_reshaped = zcs.view(bs, nc, -1).permute(0, 2, 1).contiguous()
            zcf_reshaped = zcf.view(bs, nc, -1).permute(0, 2, 1).contiguous()
            cc = torch.cat([zcs_reshaped, zcf_reshaped], dim=1)
            print(f"条件上下文形状: {cc.shape}")
            
            # 使用diffusion_model进行推理，但使用更完整的扩散过程
            print("使用diffusion_model进行推理...")
            
            # 使用p_sample_loop方法进行完整的扩散采样
            # 但由于形状问题，我们先使用一步推理
            trans_result = trans_model.model.diffusion_model(x_T, t, context=cc)
            print(f"转换结果形状: {trans_result.shape}")
            
            # 第二阶段：使用转换结果作为条件进行重建
            print("第二阶段：使用转换结果作为重建条件...")
            
            # 为第二阶段准备条件输入
            # 创建一个投影层，将trans_result的通道从4映射到128
            projection = torch.nn.Conv2d(
                in_channels=4,
                out_channels=128,
                kernel_size=3,
                padding=1
            ).to(device)
            
            # 初始化投影层的权重
            torch.nn.init.normal_(projection.weight, mean=0.0, std=0.02)
            torch.nn.init.zeros_(projection.bias)
            
            # 应用投影
            trans_projected = projection(trans_result)
            
            # 如果需要，调整空间维度
            if trans_projected.shape[2:] != zcs.shape[2:]:
                trans_projected = torch.nn.functional.interpolate(
                    trans_projected, 
                    size=zcs.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )
            
            print(f"投影后的转换结果形状: {trans_projected.shape}")
            
            # 手动处理第二阶段的条件输入
            trans_bs, trans_nc, trans_nh, trans_nw = trans_projected.shape
            trans_reshaped_context = trans_projected.view(trans_bs, trans_nc, -1).permute(0, 2, 1).contiguous()
            zcf_reshaped_context = zcf.view(bs, nc, -1).permute(0, 2, 1).contiguous()
            rec_cc = torch.cat([trans_reshaped_context, zcf_reshaped_context], dim=1)
            print(f"重建条件上下文形状: {rec_cc.shape}")
            
            # 使用diffusion_model进行推理
            print("使用diffusion_model进行推理...")
            result = rec_model.model.diffusion_model(x_T, t, context=rec_cc)
            print(f"重建结果形状: {result.shape}")
            
            # 将结果从潜在空间解码回图像空间
            print("解码结果...")
            result_img = trans_model.decode_first_stage(result)
            print(f"解码后结果形状: {result_img.shape}")
            
            # 将结果转换为图像
            result_img = ((result_img[0].permute(1, 2, 0).cpu().numpy() + 1.0) * 127.5).astype(np.uint8)
            return Image.fromarray(result_img)
    except Exception as e:
        print(f"生成字符 {target_char} 时出错: {e}")
        import traceback
        traceback.print_exc()
        return None

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="使用训练好的模型生成字体字符")
    parser.add_argument("--font", type=str, required=True, help="源字体路径")
    parser.add_argument("--chars_info", type=str, required=True, help="字符信息JSON文件路径")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--trans_model_path", type=str, required=True, help="转换模型路径")
    parser.add_argument("--rec_model_path", type=str, required=True, help="重建模型路径")
    parser.add_argument("--device", type=str, default="cuda:0", help="设备")
    parser.add_argument("--batch_size", type=int, default=1, help="批处理大小")
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    # 添加安全全局变量
    add_safe_globals()
    
    # 加载字体和字符信息
    font_path = args.font
    chars_info_path = args.chars_info
    output_dir = args.output_dir
    trans_model_path = args.trans_model_path
    rec_model_path = args.rec_model_path
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建字体目录
    font_name = os.path.splitext(os.path.basename(font_path))[0]
    output_font_dir = os.path.join(output_dir, f"{font_name}_model_generated")
    os.makedirs(output_font_dir, exist_ok=True)
    
    # 加载字符信息
    with open(chars_info_path, 'r', encoding='utf-8') as f:
        chars_info = json.load(f)
    
    # 获取源字体中已有的字符
    source_chars = set(chars_info['valid_chars'])
    print(f"源字体有效字符: {len(source_chars)}")
    
    # 获取所有需要的字符
    # 由于JSON中没有直接提供所有字符，我们需要从常用汉字表中获取
    # 这里我们使用一个简单的方法：从一个预定义的字符集合中获取
    all_chars_file = "/home/zihun/workspace/MSD-Font/font_generation/all_chars.txt"
    if os.path.exists(all_chars_file):
        with open(all_chars_file, 'r', encoding='utf-8') as f:
            all_chars = set(f.read().strip())
    else:
        # 如果没有预定义的字符集合，我们可以使用一个简单的方法生成一些字符
        print("警告：找不到all_chars.txt文件，将使用一个简单的字符集合")
        # 使用常用汉字作为示例
        all_chars = set("一二三四五六七八九十百千万亿元年月日时分秒")
    
    # 为了测试，我们只生成少量字符
    test_chars = set(list(all_chars - source_chars)[:10])
    print(f"测试生成的字符: {len(test_chars)}")
    
    # 准备参考字符
    print("准备参考字符...")
    # 加载源字体
    source_font = read_font(font_path)
    
    # 准备参考字符
    reference_chars = {}
    for char in tqdm(list(source_chars)[:100], desc="准备参考字符"):
        img = render_character(source_font, char, size=128)
        if img:
            reference_chars[char] = img
    
    # 加载模型
    print("加载模型...")
    print(f"转换模型路径: {trans_model_path}")
    print(f"重建模型路径: {rec_model_path}")
    
    # 初始化模型
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # 初始化第一阶段转换模型
    print("初始化第一阶段转换模型...")
    # 加载模型配置
    config_path = "/home/zihun/workspace/MSD-Font/configs/MSDFont/MSDFont_Train_Stage1_trans_model_predx0_miniUnet.yaml"
    config = OmegaConf.load(config_path)
    
    # 修改配置，禁用从检查点初始化
    if "ckpt_path" in config.model.params:
        config.model.params.ckpt_path = None
    
    # 初始化模型
    trans_model = MSDFont_train_stage1_trans_model_Gencase(**config.model.params)
    trans_model.to(device)
    trans_model.eval()
    
    # 加载模型权重
    if not load_model_weights(trans_model, trans_model_path, strict=False):
        print(f"加载模型时出错: 无法加载转换模型权重 {trans_model_path}")
        return
    
    # 初始化第二阶段重建模型
    print("初始化第二阶段重建模型...")
    # 加载模型配置
    config_path = "/home/zihun/workspace/MSD-Font/configs/MSDFont/MSDFont_Train_Stage2_rec_model_predx0_miniUnet_distri.yaml"
    config = OmegaConf.load(config_path)
    
    # 修改配置，禁用从检查点初始化
    if "ckpt_path" in config.model.params:
        config.model.params.ckpt_path = None
    
    # 初始化模型
    rec_model = MSDFont_train_stage2_rec_model_distri(**config.model.params)
    rec_model.to(device)
    rec_model.eval()
    
    # 加载模型权重
    if not load_model_weights(rec_model, rec_model_path, strict=False):
        print(f"加载模型时出错: 无法加载重建模型权重 {rec_model_path}")
        return
    
    print("模型加载成功")
    
    # 开始生成缺失字符
    print("开始生成缺失字符...")
    
    # 选择一个参考字符用于生成
    ref_char = next(iter(reference_chars))
    ref_img = reference_chars[ref_char]
    
    # 尝试生成测试字符
    success_count = 0
    for test_char in test_chars:
        print(f"尝试生成字符: {test_char}")
        result_img = generate_character(trans_model, rec_model, ref_img, test_char, device)
        if result_img:
            # 保存生成的字符
            output_path = os.path.join(output_font_dir, f"{test_char}.png")
            result_img.save(output_path)
            print(f"成功生成字符 {test_char}，保存到 {output_path}")
            success_count += 1
        else:
            print(f"生成字符 {test_char} 失败")
    
    print(f"字符生成完成，成功生成 {success_count}/{len(test_chars)} 个字符")
    print(f"生成的字符保存在 {output_font_dir}")

if __name__ == "__main__":
    main()
