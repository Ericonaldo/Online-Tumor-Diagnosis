# coding=utf-8
from werkzeug.utils import secure_filename
from flask import Flask, render_template, jsonify, request, make_response, send_from_directory, abort
import time
import os
import base64
import datetime
import random
from flask_cors import CORS

app = Flask(__name__)

#跨域请求
CORS(app,resources=r'/*')
#上传文件路径
UPLOAD_FOLDER = 'upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
basedir = os.path.abspath(os.path.dirname(__file__))
#允许上传的图片格式 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'gif', 'GIF'])
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS
#生成唯一的图片的名称字符串，防止图片显示时的重名问题
def create_uuid(): 
    nowTime = datetime.datetime.now().strftime("%Y%m%d%H%M%S")  # 生成当前时间
    randomNum = random.randint(0, 100)  # 生成的随机整数n，其中0<=n<=100
    if randomNum <= 10:
        randomNum = str(0) + str(randomNum)
    uniqueNum = str(nowTime) + str(randomNum)
    return uniqueNum
def test():
    arr = ['1.jpg','2.jpg','3.jpg','4.jpg','5.jpg']
    urls = []
    for item in arr:
        urls.append('http://127.0.0.1:8080/show/'+item)
    return jsonify({'data':urls})
# 上传文件
@app.route('/up_photo', methods=['POST'], strict_slashes=False)
def api_upload():
    file_dir = os.path.join(basedir, app.config['UPLOAD_FOLDER'])
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    f = request.files['photo']
    if f and allowed_file(f.filename):
        fname = secure_filename(f.filename)
        print (fname)
        ext = fname.rsplit('.', 1)[1]
        new_filename = create_uuid() + '.' + ext
        f.save(os.path.join(file_dir, new_filename))
 
        return jsonify({"success": 0, "msg": "上传成功",'data':{'file_path':'http://127.0.0.1:8808/show/'+new_filename,'file_name':new_filename,'local_path':file_dir}})
    else:
        return jsonify({"error": 1, "msg": "上传失败"})
#下载图片
@app.route('/download/<string:filename>', methods=['GET'])
def download(filename):
    if request.method == "GET":
        if os.path.isfile(os.path.join('upload', filename)):
            return send_from_directory('upload', filename, as_attachment=True)
        pass
# show photo
@app.route('/show/<string:filename>', methods=['GET'])
def show_photo(filename):
    file_dir = os.path.join(basedir, app.config['UPLOAD_FOLDER'])
    if request.method == 'GET':
        if filename is None:
            pass
        else:
            image_data = open(os.path.join(file_dir, '%s' % filename), "rb").read()
            response = make_response(image_data)
            response.headers['Content-Type'] = 'image/png'
            return response
    else:
        pass
if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 8808) 
