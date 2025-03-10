import base64

from flask import Flask, request, jsonify, send_file, Blueprint
import os
from inference import Predictor
from werkzeug.utils import secure_filename

from utils import RELEASED_WEIGHTS

app = Flask(__name__)

# 创建一个 Blueprint，并设置前缀为 /api/v1
api_v1 = Blueprint('api_v1', __name__, url_prefix='/image')

# 配置上传和生成图像的目录
UPLOAD_FOLDER = './upload_images'
GENERATED_FOLDER = './generated_images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['GENERATED_FOLDER'] = GENERATED_FOLDER

# 确保目录存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GENERATED_FOLDER, exist_ok=True)


# 定义响应类
class Response:
    def __init__(self, code, message, data=None):
        self.code = code
        self.message = message
        self.data = data

    def to_dict(self):
        return {
            "code": self.code,
            "message": self.message,
            "data": self.data
        }


# 图片转换接口
@api_v1.route('/transform', methods=['POST'])
def transform_image():
    # 检查是否有文件上传
    if 'file' not in request.files:
        return jsonify(Response(code=400, message="No file part").to_dict()), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify(Response(code=400, message="No selected file").to_dict()), 400

    # 检查是否有模型类型参数
    model_type = request.form.get('type')
    if model_type not in RELEASED_WEIGHTS:
        return jsonify(Response(code=400, message="Invalid model type").to_dict()), 400

    # 保存上传的文件
    filename = secure_filename(file.filename)
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(upload_path)

    try:
        # 初始化 Predictor
        predictor = Predictor(model_type, retain_color=True)

        # 生成图像
        generated_filename = f"generated_{filename}"
        generated_path = os.path.join(app.config['GENERATED_FOLDER'], generated_filename)
        predictor.transform_file(upload_path, generated_path)

        # 读取图片并编码为 Base64
        with open(generated_path, "rb") as file:
            encoded_file = base64.b64encode(file.read()).decode("utf-8")
        # 返回生成的图像文件路径
        return jsonify(Response(
            code=0,
            message="Image generated successfully",
            data={
                "file": encoded_file,  # Base64 图片数据
                "download_url": f"/image/download/{os.path.basename(generated_path)}"  # 下载链接
            }
        ).to_dict())

    except Exception as e:
        # 如果发生错误，返回错误信息
        return jsonify(Response(
            code=500,
            message=f"An error occurred: {str(e)}"
        ).to_dict()), 500


# 图片下载接口
@api_v1.route("/download/<filename>", methods=["GET"])
def download_file(filename):
    """提供图片下载"""
    file_path = os.path.join(app.config['GENERATED_FOLDER'], filename)
    if os.path.exists(file_path):
        return send_file(file_path, mimetype="image/png", as_attachment=True, download_name=filename)
    else:
        return jsonify({"code": 1, "message": "File not found"}), 404


if __name__ == '__main__':
    app.register_blueprint(api_v1)
    app.run(debug=True)
