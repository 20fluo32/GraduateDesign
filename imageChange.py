import os
from PIL import Image

# # 设置图片所在的文件夹路径
# folder_path = "./dataset_arcane/images"
#
# # 获取文件夹中的所有文件
# for filename in os.listdir(folder_path):
#     if filename.endswith(".png"):
#         # 拼接完整的文件路径
#         file_path = os.path.join(folder_path, filename)
#
#         # 打开图片
#         with Image.open(file_path) as img:
#             # 转换图片大小为256x256
#             img_resized = img.resize((256, 256))
#
#             # 设置新的文件名，改为.jpg格式
#             new_filename = os.path.splitext(filename)[0] + ".jpg"
#             new_file_path = os.path.join(folder_path, new_filename)
#
#             # 将图片保存为JPEG格式
#             img_resized.convert("RGB").save(new_file_path, "JPEG")
#
#             print(f"Converted {filename} to {new_filename}")

# 设置文件夹路径
folder_path = "./dataset_arcane/images"

# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    if filename.endswith(".png"):
        # 拼接文件的完整路径
        file_path = os.path.join(folder_path, filename)

        # 删除文件
        os.remove(file_path)
        print(f"Deleted {filename}")