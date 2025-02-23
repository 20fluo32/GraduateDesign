import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from scipy.linalg import sqrtm
from sklearn.decomposition import PCA
from PIL import Image
import os


# 加载预训练的 VGG16 模型（用于提取风格特征）
vgg_model = models.vgg16(pretrained=True).features
vgg_model.eval()

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def gram_matrix(features):
    """计算 Gram 矩阵"""
    batch_size, channels, height, width = features.size()
    features = features.view(batch_size * channels, height * width)
    gram = torch.mm(features, features.t())
    return gram / (channels * height * width)


def extract_style_features(image_paths, model, transform, batch_size=32, layer_idx=21):
    """提取风格特征（Gram 矩阵），支持批量处理"""
    style_features = []
    hook_fn = None

    # 定义钩子函数，用于提取指定层的特征
    def hook(module, input, output):
        nonlocal hook_fn
        hook_fn = gram_matrix(output)

    # 注册钩子
    handle = model[layer_idx].register_forward_hook(hook)

    # 按批次处理图像
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []

        # 处理当前批次的图像
        for path in batch_paths:
            img = Image.open(path).convert('RGB')
            img = transform(img).unsqueeze(0)  # 增加 batch 维度
            batch_images.append(img)

        batch_images = torch.cat(batch_images, dim=0)

        with torch.no_grad():
            model(batch_images)  # 前向传播，触发钩子函数

        style_features_batch = [hook_fn.squeeze().numpy() for _ in range(len(batch_paths))]
        style_features.extend(style_features_batch)

    # 移除钩子
    handle.remove()
    return np.array(style_features)


def calculate_fid(real_features, generated_features, n_components=500, batch_size=32):
    """计算 FID，支持批量计算"""
    # 将特征展平为二维数组
    real_features = real_features.reshape(real_features.shape[0], -1)
    generated_features = generated_features.reshape(generated_features.shape[0], -1)

    # 确保 n_components 不超过样本数和特征数中的较小值
    n_samples, n_features = real_features.shape
    n_components = min(n_components, n_samples, n_features)

    # 使用 PCA 降维
    pca = PCA(n_components=n_components)
    real_features_pca = pca.fit_transform(real_features)
    generated_features_pca = pca.transform(generated_features)

    # 计算均值和协方差矩阵
    mu1, sigma1 = real_features_pca.mean(axis=0), np.cov(real_features_pca, rowvar=False)
    mu2, sigma2 = generated_features_pca.mean(axis=0), np.cov(generated_features_pca, rowvar=False)

    diff = mu1 - mu2
    covmean = sqrtm(sigma1.dot(sigma2))

    # 检查 sqrtm 的结果是否为复数
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid



# 示例：计算风格FID
real_image_paths = [os.path.join('style_images', f) for f in os.listdir('style_images')]
generated_image_paths = [os.path.join('generated_images', f) for f in os.listdir('generated_images')]

# 提取风格特征
real_style_features = extract_style_features(real_image_paths, vgg_model, transform, batch_size=32)
generated_style_features = extract_style_features(generated_image_paths, vgg_model, transform, batch_size=32)

# 计算风格FID
style_fid = calculate_fid(real_style_features, generated_style_features, n_components=500, batch_size=32)
print(f"Style FID: {style_fid}")
