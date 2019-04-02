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
import matplotlib.image as mpimg

import pandas as pd
from pathlib import Path
from torch.nn import DataParallel

from sklearn.model_selection import train_test_split
from skimage.transform import resize

from fastai.callbacks.hooks import *

app = Flask(__name__)

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
    pred = '%.2f' % pred*100
    return render_template('input_id.html', filenames=filenames, pred=pred)

def get_filenames_org(_id):
    return [f'{_id}_{i}' for i in range(21)]

def get_pred(filename):
    return
