import torch
import torchvision.models as models
import torchvision.transforms as transforms
from scipy.linalg import sqrtm
import numpy as np
from PIL import Image
import os

# 加载预训练的InceptionV3模型
inception_model = models.inception_v3(pretrained=True, transform_input=False)
inception_model.fc = torch.nn.Identity()  # 去掉最后的全连接层，直接输出特征
inception_model.eval()

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def extract_features(image_paths, model, transform):
    features = []
    for path in image_paths:
        img = Image.open(path).convert('RGB')
        img = transform(img).unsqueeze(0)
        with torch.no_grad():
            feat = model(img)
        features.append(feat.squeeze().numpy())
    return np.array(features)


def calculate_fid(real_features, generated_features):
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = generated_features.mean(axis=0), np.cov(generated_features, rowvar=False)
    diff = mu1 - mu2
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid


# 计算语义FID
real_image_paths = [os.path.join('real_images', f) for f in os.listdir('real_images')]
generated_image_paths = [os.path.join('generated_images', f) for f in os.listdir('generated_images')]

real_features = extract_features(real_image_paths, inception_model, transform)
generated_features = extract_features(generated_image_paths, inception_model, transform)

semantic_fid = calculate_fid(real_features, generated_features)
print(f"Semantic FID: {semantic_fid}")
