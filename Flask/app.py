from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import pickle

# Load the pre-trained model and tokenizer
# Replace 'model.h5' and 'tokenizer.pkl' with your actual file paths
model = tf.keras.models.load_model('./Model/model_predictor.h5')
with open('./Model/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

app = Flask(__name__)

def predict_next_word(input_text):
    # Tokenize the input text
    sequence = tokenizer.texts_to_sequences([input_text])
    sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=model.input_shape[1])

    # Predict the next word
    predictions = model.predict(sequence)
    predicted_index = np.argmax(predictions)

    # Map the predicted index to a word
    reverse_word_map = {v: k for k, v in tokenizer.word_index.items()}
    next_word = reverse_word_map.get(predicted_index, "<unknown>")

    return next_word

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form['user_input']
    next_word = predict_next_word(user_input)
    return jsonify({'next_word': next_word})

if __name__ == '__main__':
    app.run(debug=True)
