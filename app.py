import os
import joblib
import re
import PyPDF2
from flask import Flask, render_template, request

# Load the pre-trained models
clf = joblib.load('clf.pkl')
tfidf = joblib.load('tfidf.pkl')

def extract_text_from_pdf(pdf_path):
    pdf_text = ""
    try:
        with open(pdf_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            for page in pdf_reader.pages:
                pdf_text += page.extract_text()
    except Exception as e:
        return str(e)
    return pdf_text

def cleanResume(txt):
    # Your cleanResume function here
    cleanText = re.sub(r'http\S+\s', ' ', txt)
    cleanText = re.sub(r'RT|cc', ' ', cleanText)
    cleanText = re.sub(r'#\S+\s', ' ', cleanText)
    cleanText = re.sub(r'@\S+', '  ', cleanText)
    cleanText = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub(r'\s+', ' ', cleanText)
    return cleanText

# Category mapping here
category_mapping = {
    15: "Java Developer",
    23: "Testing",
    8: "DevOps Engineer",
    20: "Python Developer",
    24: "Web Designing",
    12: "HR",
    13: "Hadoop",
    3: "Blockchain",
    10: "ETL Developer",
    18: "Operations Manager",
    6: "Data Science",
    22: "Sales",
    16: "Mechanical Engineer",
    1: "Arts",
    7: "Database",
    11: "Electrical Engineering",
    14: "Health and fitness",
    19: "PMO",
    4: "Business Analyst",
    9: "DotNet Developer",
    2: "Automation Testing",
    17: "Network Security Engineer",
    21: "SAP Developer",
    5: "Civil Engineer",
    0: "Advocate",
}


app = Flask(__name__)

# Define the directory where uploaded files will be stored
UPLOAD_FOLDER = 'uploads'

# Set the configuration for the upload folder
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the "uploads" directory exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])



@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/serv')
def serv():
    return render_template('service.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            # Save the uploaded resume
            resume_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
            uploaded_file.save(resume_path)

            # Extract text from the uploaded resume
            text = extract_text_from_pdf(resume_path)

            # Clean the input resume
            cleaned_resume = cleanResume(text)

            # Transform the cleaned resume using the TfidfVectorizer
            input_features = tfidf.transform([cleaned_resume])

            # Make the prediction using the classifier
            prediction_id = clf.predict(input_features)[0]

            # Map category ID to category name
            category_name = category_mapping.get(prediction_id, "Unknown")

            return render_template('result.html', category=category_name)

        else:
            return render_template('service.html', error='Please upload a resume file.')

if __name__ == '__main__':
    app.run(debug=True)
