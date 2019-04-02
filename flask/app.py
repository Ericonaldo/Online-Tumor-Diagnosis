from flask import Flask, render_template, Response, request, redirect, url_for, send_file
from flask_wtf import Form
from wtforms import StringField
from wtforms.validators import DataRequired
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import io
import PIL
from pathlib import Path
import os
from fastai.vision import *
from skimage.transform import resize
import skimage.io
import json
import jsonify
import matplotlib.image as mpimg
import dicom

import pandas as pd
from pathlib import Path
from torch.nn import DataParallel

from sklearn.model_selection import train_test_split
from skimage.transform import resize

from fastai.callbacks.hooks import *

app = Flask(__name__)
#跨域请求
CORS(app,resources=r'/*')
#上传文件路径
UPLOAD_FOLDER = 'upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
basedir = os.path.abspath(os.path.dirname(__file__))
#允许上传的图片格式 
ALLOWED_EXTENSIONS = set(['dcm'])


train2 = pd.read_csv('train2.csv')[lambda x: ((x.ymax - x.ymin) <= (x.xmax - x.xmin)) & (x.xmin != 200)]
train2['filename'] = '/mnt/bak/ga/train2/' + train2.index.astype(str) + '_2.npy'
train2 = train2.reset_index(drop=True)

BS_SIZE = 32
NUM_WORKERS = 8

train_idx, val_idx = train_test_split(range(len(train2)), test_size=0.2, random_state=666, stratify=train2.ret)

def open_npy(self, fname):
    x = np.load(fname).astype(np.float32) / 255
    x = torch.from_numpy(x)
    return Image(x)
ImageList.open=open_npy
tfms = get_transforms(do_flip=False)
data = (ImageList.from_df(path='/', df=train2, cols='filename')
    .split_by_idxs(train_idx, val_idx)
    .label_from_df('ret')
    .add_test(ImageList.from_df(path='/', df=train2, cols='filename'))
    .transform(tfms, size=(21, 320, 400), padding_mode='zeros')
    .databunch(bs=BS_SIZE, num_workers=NUM_WORKERS)
    .normalize())

class Dummy(nn.Module):
    "Just returns the mean of the `output`."
    def forward(self, output, *args): return output

# resnet34
class CombineAllChannelsNet(nn.Module):
    def __init__(self):
        super(CombineAllChannelsNet, self).__init__()
        body = create_body(models.resnet34)
        body_out_features = 512
        body[0].weight.data = body[0].weight.data[:, 0].unsqueeze(1)
        body[0].in_channels = 1
        self.body = body
        self.neck = Dummy()
        self.head = create_head(21*body_out_features*2, 2, [1024])  # 之前的输出是512，21个channels，每个有两个输出，所以是21*1024

    def forward(self, inp):
        lst_oup = []
        for i in range(21):
            lst_oup.append(self.body(inp[:, i].unsqueeze(1)))
        combined = torch.cat(lst_oup, dim=1)
        combined = self.neck(combined)
        out = self.head(combined)
        return out

name = f'/mnt/bak/ga/model/combinenet_all_tfms_unfreeze'
model = CombineAllChannelsNet()
learn = Learner(data, model, metrics=[accuracy])
learn.load(name);
m = learn.model.eval();

@app.route('/')
def home_page():
    return redirect(url_for('input_id'))

@app.route('/input_id/', methods=['GET', 'POST'])
def input_id():
    if request.method == 'POST':
        return redirect(url_for('plot', _id=request.form['txt_id']))
    else:
        return render_template('input_id.html')

@app.route('/<_id>')
def plot(_id):
    # delete images
    for p in Path('./static/').glob('*.jpg'):
        os.remove(p)
    # generate images
    filename = f'/mnt/bak/ga/train2/{_id}_2.npy'
    xb, _ = data.one_item(open_npy(None, filename))
    xb_im = Image(data.denorm(xb)[0])
    xb = xb.cuda()
    yb = Tensor([1]).cuda()  # 如果是gracam，这里要改成真实的标签
    pred = learn.pred_batch(DatasetType.Valid, (xb, yb)).numpy()[0, 1]
    
    imgs = np.load(filename)
    filenames = get_filenames_org(_id)
    for img, filename in zip(imgs, filenames):
        PIL.Image.fromarray(img).save(f'./static/{filename}.jpg', 'JPEG', quality=90)
    def hooked_backward(cat):
        with hook_output(m.neck) as hook_a: 
            with hook_output(m.neck, grad=True) as hook_g:
                preds = m(xb)
                preds[0,int(cat)].backward()
        return hook_a,hook_g

    hook_a,hook_g = hooked_backward(1)  # 如果是gracam，这里要改成真实的标签
    acts = hook_a.stored[0].cpu()
    for i, filename in enumerate(filenames):
        hm = acts[i*512:(i+1)*512].mean(0)
        hm_resized = resize(hm.numpy(), (320, 400))
        hm_resized = (hm_resized - hm_resized.min()) / (hm_resized.max() - hm_resized.min())
        skimage.io.imsave(f'./static/{filename}_hm.jpg', hm_resized, quality=90)
    return render_template('images.html', filenames=filenames, pred=pred)

def get_filenames_org(_id):
    return [f'{_id}_{i}' for i in range(21)]

def get_pred(filename):
    return

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
 
        return jsonify({"success": 0, "msg": "上传成功",'data':{'file_path':'106.12.11.68:8808/show/'+new_filename,'file_name':new_filename,'local_path':file_dir}})
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

def input_process(dcm_dir, dcm_name):
    # read dcm
    dcm = dicom.read_file(dcm_dir+dcm+name)
    # TODO here shold be pre process codes
    # TODO here should be codes that does inference
    xb_im = Image(img)
    xb = xb.cuda()
    yb = Tensor([1]).cuda()  # 如果是gracam，这里要改成真实的标签
    pred = learn.pred_batch(DatasetType.Valid, (xb, yb)).numpy()[0, 1]
    
    # TODO here should be codes that outputs the heat maps

    data = {
    "a": "Runoob",
    "b": 7
    }
    data = json.dumps(data)

    return data

if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 8808)  
