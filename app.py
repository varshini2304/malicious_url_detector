from flask import Flask, request, jsonify
import torch
from models.transformer import TransformerClassifier
from utils.preprocessing import build_char_to_idx

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load("malicious_url_detector/saved_model/transformer_model.pth", map_location=device)
model = TransformerClassifier(
    vocab_size=len(checkpoint['char2idx']) + 1,
    d_model=128,
    num_heads=4,
    d_ff=512,
    num_layers=2,
    num_classes=2,
    dropout=0.1,
    max_len=200
)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()
char2idx = checkpoint['char2idx']

@app.route('/')
def home():
    return '''
    <h2>Enter a URL to Check if it's Malicious</h2>
    <form method="POST" action="/predict">
        <input type="text" name="url" placeholder="https://example.com" size="50">
        <input type="submit" value="Check">
    </form>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    url = request.form.get('url')
    if not url:
        return jsonify({'error': 'No URL provided'}), 400
    encoded = [char2idx.get(c, 0) for c in url.lower()]
    max_len = 200
    if len(encoded) > max_len:
        encoded = encoded[:max_len]
    else:
        encoded += [0] * (max_len - len(encoded))
    input_tensor = torch.tensor([encoded]).to(device)
    with torch.no_grad():
        output = model(input_tensor)
    pred_class = torch.argmax(output, dim=1).item()
    class_names = {0: "BENIGN", 1: "MALICIOUS"}
    return jsonify({'url': url, 'prediction': class_names.get(pred_class, 'UNKNOWN'), 'class': pred_class})

if __name__ == '__main__':
    app.run(debug=True)
