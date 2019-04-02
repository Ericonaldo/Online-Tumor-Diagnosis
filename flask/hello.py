from flask import Flask, render_template, Response, request, redirect, url_for
from flask_wtf import Form
from wtforms import StringField
from wtforms.validators import DataRequired
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import io

app = Flask(__name__)

@app.route('/')
def hello_world():
    return '戴玮是傻鸟'

@app.route('/lmh/')
def lmh():
    return render_template('lmh.html')

@app.route('/get_prob/<_id>')
def get_prob(_id):
    return f'{_id}的概率是:'

@app.route('/input_id/', methods=['GET', 'POST'])
def input_id():
    if request.method == 'POST':
        return redirect(url_for('plot_png', _id=request.form['txt_id']))
    else:
        return render_template('input_id.html')

@app.route('/<_id>')
def plot_png(_id):
    fig = create_figure(_id)
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

def create_figure(_id):
    fig = Figure()
    ax = fig.add_subplot(111)
    img = np.load(f'/mnt/bak/ga/train2/{_id}_2.npy')[11]
    ax.imshow(img, cmap='gray')
    return fig
