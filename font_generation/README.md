# MSD-Font 字体生成工具

这个目录包含了使用 MSD-Font 模型从有限字符集的字体生成完整字库的工具。

## 文件说明

- `create_char_list.py`: 创建完整的 6763 个汉字字符列表
- `generate_full_font.sh`: 使用 MSD-Font 模型生成完整字库的 Shell 脚本
- `generate_full_font_batched.py`: 使用 MSD-Font 模型批量生成完整字库的 Python 脚本（推荐使用）

## 使用方法

### 1. 准备字符列表

首先，生成完整的汉字字符列表：

```bash
cd /home/zihun/workspace/MSD-Font/font_generation
python create_char_list.py
```

这将生成两个文件：
- `all_chars.txt`: 包含所有汉字的文本文件
- `all_chars.json`: 包含所有汉字的 JSON 文件

### 2. 使用批量生成脚本

使用 Python 脚本批量生成字体（推荐）：

```bash
python generate_full_font_batched.py --source_font /path/to/your/font.ttf --output_dir ./output
```

参数说明：
- `--source_font`: 源字体文件路径（必需）
- `--output_dir`: 输出目录（默认为 ./output）
- `--char_list`: 字符列表文件路径（可选，默认使用内置的 6763 个汉字）
- `--batch_size`: 批处理大小（默认为 16）

### 3. 使用 Shell 脚本

或者，您也可以使用 Shell 脚本：

```bash
bash generate_full_font.sh /path/to/your/font.ttf
```

## 注意事项

1. 源字体文件应该至少包含几百个字符，以确保生成的字体质量
2. 生成过程可能需要较长时间，请耐心等待
3. 生成的字体将保存在指定的输出目录中
4. 生成过程使用 GPU 加速，请确保您的系统有可用的 CUDA 设备

## 高级选项

如果您需要更多控制，可以直接修改 Python 脚本中的参数：

- 调整批处理大小以适应您的 GPU 内存
- 修改模型路径以使用不同的预训练模型
- 自定义字符列表以生成特定的字符集
