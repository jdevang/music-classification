from flask import Flask, render_template, flash, request, redirect, url_for
from model import predictor
from werkzeug.utils import secure_filename
import os, math


UPLOAD_FOLDER = 'songs/'
ALLOWED_EXTENSIONS = { 'mp3', 'wav' }

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
  if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # x = ("classical", "kek", [3,2,4,1,6,0,7,5,8])
            classes = {
                "blues", 
                "classical",
                "country",
                "disco",
                "hip-hop",
                "jazz",
                "metal",
                "pop",
                "reggae",
                "rock"
            }
            x = predictor(filename)
            return render_template('index.html', x=x, prediction=x[0], probability=x[2].tolist(), classes=classes, filename=file.filename, output=True)
  return render_template('index.html')


app.jinja_env.globals.update(sorted=sorted)
app.jinja_env.globals.update(zip=zip)

if __name__ == '__main__':
  app.run(host='127.0.0.1', port=8000, debug=True)
 