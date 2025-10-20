import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse
import os
import json
import random
import re
from model import CSRNet

def main():
    """
    该脚本用于对一个图像列表进行批量多帧输入的人群计数推理，
    并随机生成一部分样本的可视化对比图。
    """
    # --- 1. 定义命令行参数 ---
    parser = argparse.ArgumentParser(description='CSRNet Multi-Frame Batch Inference and Visualization')
    parser.add_argument('--model_path', type=str, required=True,
                        help='训练好的9通道模型检查点路径 (.pth.tar)')
    parser.add_argument('--input_json', type=str, required=True,
                        help='包含图片路径列表的JSON文件 (例如: test.json 或 Val.json)')
    parser.add_argument('--output_dir', type=str, default='./batch_inference_results',
                        help='保存可视化对比图的目录')
    parser.add_argument('--num_visualizations', type=int, default=10,
                        help='要随机生成的可视化样本数量')
    args = parser.parse_args()

    # --- 2. 环境设置和目录创建 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"✅ 推理设备: {device}")
    print(f"✅ 可视化结果将保存至: {args.output_dir}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # --- 3. 加载模型 ---
    print("\n🚀 正在加载模型...")
    model = CSRNet().to(device)
    try:
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        print(f"✅ 模型成功从 '{args.model_path}' 加载。")
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        return

    # --- 4. 加载图片列表 ---
    try:
        with open(args.input_json, 'r') as f:
            image_paths = json.load(f)
        print(f"✅ 成功从 '{args.input_json}' 加载 {len(image_paths)} 张图片路径。")
    except Exception as e:
        print(f"❌ 加载JSON文件失败: {e}")
        return

    # --- 5. 批量推理循环 ---
    results = []
    print("\n🧠 开始批量推理...")
    for i, center_img_path in enumerate(image_paths):
        try:
            # 解析文件名
            match = re.search(r'(\d+)_(\d+)\.jpg', os.path.basename(center_img_path))
            if not match:
                print(f"  - 警告: 跳过不符合格式的文件: {os.path.basename(center_img_path)}")
                continue
            
            seq_id, frame_num_str = match.groups()
            frame_num = int(frame_num_str)

            # 构建相邻帧路径
            dir_name = os.path.dirname(center_img_path)
            prev_frame_path = os.path.join(dir_name, f"{seq_id}_{frame_num - 1:05d}.jpg")
            next_frame_path = os.path.join(dir_name, f"{seq_id}_{frame_num + 1:05d}.jpg")

            # 处理边界情况
            prev_frame_path = prev_frame_path if os.path.exists(prev_frame_path) else center_img_path
            next_frame_path = next_frame_path if os.path.exists(next_frame_path) else center_img_path
            
            # 加载三张图片
            images = [Image.open(p).convert('RGB') for p in [prev_frame_path, center_img_path, next_frame_path]]
            
            # 转换为张量并堆叠
            images_tensor = [transform(img) for img in images]
            stacked_tensor = torch.cat(images_tensor, dim=0).unsqueeze(0).to(device)
            
            # 推理
            with torch.no_grad():
                output = model(stacked_tensor)
            
            count = output.detach().cpu().sum().item()
            print(f"  [{i+1}/{len(image_paths)}] 图片: {os.path.basename(center_img_path)}, 预测人数: {count:.2f}")

            results.append({
                'path': center_img_path,
                'count': count,
                'density_map': output.detach().cpu().squeeze().numpy()
            })
        except Exception as e:
            print(f"  - 错误: 处理图片 {os.path.basename(center_img_path)} 时失败: {e}")

    print("✅ 批量推理完成！")

    # --- 6. 随机可视化 ---
    if not results:
        print("\n❌ 没有成功的推理结果，无法生成可视化。")
        return

    num_to_visualize = min(args.num_visualizations, len(results))
    print(f"\n🖼️  正在随机生成 {num_to_visualize} 张可视化对比图...")
    
    random_samples = random.sample(results, num_to_visualize)

    for sample in random_samples:
        img_raw = Image.open(sample['path']).convert('RGB')
        
        plt.figure(figsize=(20, 10))
        
        # 原图
        plt.subplot(1, 2, 1)
        plt.imshow(img_raw)
        plt.title('Original Image')
        plt.axis('off')
        
        # 预测密度图
        plt.subplot(1, 2, 2)
        plt.imshow(sample['density_map'], cmap=cm.jet)
        plt.title(f"Predicted Density Map\nCount: {sample['count']:.2f}")
        plt.axis('off')
        
        # 保存图像
        output_filename = os.path.join(args.output_dir, f"vis_{os.path.basename(sample['path'])}")
        plt.savefig(output_filename, bbox_inches='tight')
        plt.close() # 释放内存
        print(f"  - 已保存对比图: {os.path.basename(output_filename)}")

    print("🎉 所有任务完成！")

if __name__ == '__main__':
    main()
