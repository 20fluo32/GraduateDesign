import onnxruntime as ort
import numpy as np
import torch
import cv2
from torchvision import transforms
from PIL import Image

# 1. 加载 ONNX 模型
onnx_path = "./exports/generator_v2.onnx"  # 替换为你的 ONNX 模型路径
session = ort.InferenceSession(onnx_path)


# 2. 准备输入数据
def preprocess_image(image_path):
    """
    预处理函数：将输入图像转换为模型所需的格式，但不调整大小。
    """
    # 读取图像
    image = Image.open(image_path).convert("RGB")
    # 转换为张量并归一化到 [-1, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    image_tensor = transform(image).unsqueeze(0)  # 添加 batch 维度
    return image_tensor


# 4. 后处理输出
def postprocess_output(output_tensor, original_size):
    """
    后处理函数：将模型输出转换为图像，并调整回原始大小。
    """
    # 将输出张量从 [-1, 1] 反归一化到 [0, 1]
    output_tensor = (output_tensor + 1) / 2.0
    # 将张量转换为 PIL 图像
    output_image = transforms.ToPILImage()(output_tensor.squeeze(0))
    # 调整回原始大小
    output_image = output_image.resize(original_size, Image.BILINEAR)
    return output_image


# 输入图像路径
input_image_path = "./temp.jpg"  # 替换为你的输入图像路径
input_image = Image.open(input_image_path).convert("RGB")
original_size = input_image.size  # 获取原始图像大小

# 预处理输入图像
input_tensor = preprocess_image(input_image_path)

# 将 PyTorch 张量转换为 NumPy 数组
input_data = input_tensor.numpy()

# 运行推理
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
output_data = session.run([output_name], {input_name: input_data})[0]

# 后处理输出
output_image = postprocess_output(torch.from_numpy(output_data[0]), original_size)

# 保存输出图像
output_image_path = "./anime.jpg"
output_image.save(output_image_path)
print(f"生成的图片已保存到 {output_image_path}")

# # 显示输入和输出图像
# input_image = cv2.cvtColor(cv2.imread(input_image_path), cv2.COLOR_BGR2RGB)
# output_image = cv2.cvtColor(cv2.imread(output_image_path), cv2.COLOR_BGR2RGB)
#
# cv2.imshow("Input Image", input_image)
# cv2.imshow("Output Image", output_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
