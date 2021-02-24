import os

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from models import predict_animal

app = Flask(__name__)

UPLOAD_FOLDER = './static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = os.urandom(24)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        upload_img= request.files['file']
        filename = secure_filename(upload_img.filename)
        upload_img.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        img_url = '/static/' + filename
        if not upload_img:
            return render_template('index.html')
        else:
            return render_template('confirm.html', upload_file=img_url)
    else:
        return render_template('index.html')

@app.route('/v1/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        pred_file = request.form['image']
        result = predict_animal(pred_file)
    return render_template('result.html', upload_file=pred_file, result= result)
        

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000, debug=True)