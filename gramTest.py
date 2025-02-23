import torch
import torchvision
import torchvision.transforms as transforms
from pytorch_fid import fid_score

# 准备真实数据分布和生成模型的图像数据
real_images_folder = 'real_images'
generated_images_folder = 'generated_images'

# 加载预训练的Inception-v3模型，并提取特征提取部分
inception_model = torchvision.models.inception_v3(pretrained=True, transform_input=False)
inception_model.fc = torch.nn.Identity()  # 移除分类头，只保留特征提取部分
inception_model.eval()  # 设置为评估模式

# 定义图像变换
transform = transforms.Compose([
    transforms.Resize((299, 299)),  # 将所有图像调整为299x299
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 计算FID距离值
fid_value = fid_score.calculate_fid_given_paths(
    paths=[real_images_folder, generated_images_folder],
    batch_size=50,  # 可以根据显存大小调整
    device='cuda',  # 使用GPU
    dims=2048,       # Inception-v3模型的输出特征维度
    num_workers=0    # 禁用多线程
)
print('FID value:', fid_value)