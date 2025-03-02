from flask import Flask

# 实力化对象Flask
app = Flask(__name__)


# 创建了网址/show/info和函数index的对应关系
# 以后用户再浏览器访问/show/info时，网站自动执行函数index
@app.route("/show/info")
def index():
    return "hello world"


if __name__ == "__main__":
    app.run()
