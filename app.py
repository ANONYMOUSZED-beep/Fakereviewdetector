from flask import Flask, request, render_template
import pickle
import gzip

# Load the saved model and vectorizer
with gzip.open('fake_review_model.pkl.gz', 'rb') as f:
    model = pickle.load(f)

with gzip.open('tfidf_vectorizer.pkl.gz', 'rb') as f:
    vectorizer = pickle.load(f)

# Initialize Flask app
app = Flask(__name__)

# Home route
@app.route('/')
def home():
    return render_template('htmlfake.html')

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    # Get the review from the form
    review = request.form['reviewText']

    # Preprocess and vectorize the review
    vectorized_review = vectorizer.transform([review])

    # Make a prediction
    prediction = model.predict(vectorized_review)[0]

    # Interpret prediction
    result = "Fake Review" if prediction == 1 else "Genuine Review"

    return render_template('htmlfake.html', prediction_text=f'Review Result: {result}')

if __name__ == "__main__":
    app.run(debug=True)