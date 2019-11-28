from flask import Flask, render_template, flash, request, redirect, url_for
from model import predictor
from werkzeug.utils import secure_filename
import os, boto3, json


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
            S3_BUCKET = os.environ.get('S3_BUCKET_NAME')
            filename = secure_filename(file.filename)
            filetype = file.filetype
            s3 = boto3.client('s3')
            # file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            presigned_post = s3.generate_presigned_post(
              Bucket = S3_BUCKET,
              Key = file_name,
              Fields = {"acl": "public-read", "Content-Type": file_type},
              Conditions = [
                {"acl": "public-read"},
                {"Content-Type": file_type}
              ],
              ExpiresIn = 3600
            )
            # x = predictor(filename)
            # return x
  return render_template('index.html')
if __name__ == '__main__':
  app.run(host='127.0.0.1', port=8000, debug=True)
 