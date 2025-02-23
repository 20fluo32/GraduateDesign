from inference import Predictor

# 测试代码
predictor = Predictor(
    './runs_train_photo_Hayao/GeneratorV2_train_photo_Hayao.pt',
    # './runs_train_photo_Paprika/epoch_90/GeneratorV2_train_photo_Paprika.pt',
    # './runs_train_photo_Shinkai/epoch_50/GeneratorV2_train_photo_Shinkai.pt',
    # './runs_train_photo_Arcane/GeneratorV2_train_photo_Arcane.pt',
    # 'hayao:v2',
    # if set True, generated image will retain original color as input image
    retain_color=True
)

url = 'https://github.com/ptran1203/pytorch-animeGAN/blob/master/example/result/real/1%20(20).jpg?raw=true'

predictor.transform_file(url, "./anime.jpg")
# predictor.transform_file(url, "./anime2.jpg")
