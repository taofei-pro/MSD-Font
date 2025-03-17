#!/bin/bash

# 设置环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate MSDFont

# 配置变量
SOURCE_FONT="$1"  # 源字体文件路径
OUTPUT_DIR="/home/zihun/workspace/MSD-Font/font_generation/output"
CHAR_LIST="/home/zihun/workspace/MSD-Font/font_generation/all_chars.txt"
CONFIG_REC="/home/zihun/workspace/MSD-Font/configs/MSDFont/MSDFont_Eval_rec_model_predx0_miniUnet.yaml"
CONFIG_TRANS="/home/zihun/workspace/MSD-Font/configs/MSDFont/MSDFont_Eval_trans_model_predx0_miniUnet.yaml"
REC_MODEL="/home/zihun/workspace/MSD-Font/logs/2025-03-13T12-17-14_MSDFont_Train_Stage2_rec_model_predx0_miniUnet_distri_updated/checkpoints/last.ckpt"
TRANS_MODEL="/home/zihun/workspace/MSD-Font/logs/2025-03-12T18-11-23_MSDFont_Train_Stage1_trans_model_predx0_miniUnet/checkpoints/last.ckpt"

# 创建必要的目录
mkdir -p "$OUTPUT_DIR"

# 检查源字体文件是否存在
if [ ! -f "$SOURCE_FONT" ]; then
    echo "错误：源字体文件 $SOURCE_FONT 不存在"
    exit 1
fi

# 检查字符列表是否存在，如果不存在则创建
if [ ! -f "$CHAR_LIST" ]; then
    echo "创建字符列表文件..."
    python -c "
import json
# 常用汉字6763个
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
# 保存到文件
with open('$CHAR_LIST', 'w', encoding='utf-8') as f:
    f.write(''.join(chars))
print(f'已生成 {len(chars)} 个汉字')
"
fi

# 运行生成脚本
echo "开始生成字体..."
python /home/zihun/workspace/MSD-Font/scripts/MSDFont_eval.py \
    --outdir "$OUTPUT_DIR" \
    --path_genchar "$CHAR_LIST" \
    --path_refchar "$CHAR_LIST" \
    --path_ttf "$SOURCE_FONT" \
    --source_path "$SOURCE_FONT" \
    --path_config_rec "$CONFIG_REC" \
    --path_rec_model "$REC_MODEL" \
    --path_config_trans "$CONFIG_TRANS" \
    --path_trans_model "$TRANS_MODEL"

echo "字体生成完成！输出目录: $OUTPUT_DIR"
