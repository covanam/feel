from flask import Flask, request, redirect, render_template
import os
from model import Feel
import cv2
import pickle
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'upload')
model = pickle.load(open('model/model', 'rb'))
FILENAME = ''
RESULT = 'hi'


def process():
    s = 'upload/' + FILENAME
    x = cv2.imread(s)
    x = cv2.resize(x, (224, 224))
    cv2.imwrite('static/display.png', x)
    y = model(x.astype(np.float32))
    global RESULT
    RESULT = 'countryside' if y == 1 else "metropotalian"


@app.route('/upload', methods=["POST"])
def upload():
    f = request.files['file']

    global FILENAME
    FILENAME = f.filename

    f.save(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))

    process()

    return redirect('/')


@app.route('/')
def form():
    return render_template("web.html", outcome=RESULT)


if __name__ == '__main__':
    app.run()