import torch
import argparse
import os
from models.anime_gan import GeneratorV1
from models.anime_gan_v2 import GeneratorV2
from models.anime_gan_v3 import GeneratorV3
from utils.common import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser()
    # 模型相关参数
    parser.add_argument('--model', type=str, default='v2', help="AnimeGAN version, can be {'v1', 'v2', 'v3'}")
    parser.add_argument('--resume_G', type=str, default='./runs_train_photo_Hayao/GeneratorV2_train_photo_Hayao.pt', help="Path to generator weights")
    parser.add_argument('--imgsz', type=int, nargs="+", default=[256], help="Image size for ONNX export")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help="Device to use")

    # 导出路径相关参数
    parser.add_argument('--exp_dir', type=str, default='exports', help="Directory to save ONNX model")

    # 其他可选参数
    parser.add_argument('--use_sn', action='store_true', help="Whether to use spectral normalization")
    parser.add_argument('--d_layers', type=int, default=2, help="Number of discriminator conv layers")

    return parser.parse_args()


def export_onnx(args):
    # 选择模型版本
    if args.model == 'v1':
        G = GeneratorV1()
    elif args.model == 'v2':
        G = GeneratorV2()
    elif args.model == 'v3':
        G = GeneratorV3()
    else:
        raise ValueError(f"Unsupported model version: {args.model}")

    # 加载权重
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.resume_G, map_location=device)  # 加载 checkpoint
    if 'model_state_dict' in checkpoint:
        # 如果 checkpoint 包含 "model_state_dict"，则提取模型权重
        state_dict = checkpoint['model_state_dict']
    else:
        # 否则直接使用 checkpoint 作为 state_dict
        state_dict = checkpoint

    # 加载模型权重
    G.load_state_dict(state_dict)

    # 设置模型为评估模式
    G.to(device)
    G.eval()

    # 创建一个随机输入张量，用于导出 ONNX 模型
    # 假设输入图像的大小为 [batch_size, channels, height, width]
    batch_size = 1  # ONNX 导出通常使用 batch_size=1
    channels = 3  # RGB 图像
    height, width = args.imgsz[0], args.imgsz[0]  # 假设 imgsz 是一个列表，包含图像的高度和宽度
    dummy_input = torch.randn(batch_size, channels, height, width, device=args.device)

    # 导出 ONNX 模型
    onnx_path = os.path.join(args.exp_dir, f"generator_{args.model}.onnx")
    torch.onnx.export(
        G,  # 要导出的模型
        dummy_input,  # 模型输入
        onnx_path,  # 导出的 ONNX 文件路径
        export_params=True,  # 导出模型参数
        opset_version=10,  # ONNX 操作集版本
        do_constant_folding=True,  # 是否执行常量折叠优化
        input_names=['input'],  # 输入张量的名称
        output_names=['output'],  # 输出张量的名称
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}  # 支持动态 batch size
    )

    print(f"ONNX model exported to {onnx_path}")


if __name__ == '__main__':
    args = parse_args()

    # 导出 ONNX 模型
    export_onnx(args)
