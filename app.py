from flask import Flask, request, redirect, url_for, render_template
import os

app = Flask(__name__)

# Directory for storing uploaded images
UPLOAD_FOLDER = 'images'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Function to check if the uploaded file has a valid extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index_2.html')

@app.route('/upload_file', methods=['POST'])
def upload_file():
    # Check if the file is in the request
    if 'in_img' not in request.files:
        return redirect(request.url)
    
    file = request.files['in_img']
    
    # If no file was selected or the file type is not allowed
    if file.filename == '' or not allowed_file(file.filename):
        return redirect(request.url)
    
    # Save the file in the UPLOAD_FOLDER with its original name
    filename = 'new_image.' + file.filename.rsplit('.', 1)[1].lower()  # Save as 'new_image' with original extension
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    # Save the uploaded file to the specified path
    file.save(file_path)
    
    return f"File uploaded successfully! Saved as: {file_path}"

if __name__ == '__main__':
    app.run(debug=True)
