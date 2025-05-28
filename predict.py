import torch
from models.transformer import TransformerClassifier

def load_model(path, device):
    checkpoint = torch.load(path, map_location=device)
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
    return model, checkpoint['char2idx']

import sys

def predict_url(model, char2idx, url, max_len=200, device='cpu'):
    encoded = [char2idx.get(c, 0) for c in url.lower()]
    if len(encoded) > max_len:
        encoded = encoded[:max_len]
    else:
        encoded += [0] * (max_len - len(encoded))
    input_tensor = torch.tensor([encoded]).to(device)
    with torch.no_grad():
        output = model(input_tensor)
    pred_class = torch.argmax(output, dim=1).item()
    class_names = {0: "BENIGN", 1: "MALICIOUS"}
    print(f"URL: {url}")
    print(f"Prediction: {class_names.get(pred_class, 'UNKNOWN')} (class {pred_class})")
    return pred_class

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide a URL as a command line argument.")
        sys.exit(1)
    url = sys.argv[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, char2idx = load_model("malicious_url_detector/saved_model/transformer_model.pth", device)
    predict_url(model, char2idx, url, device=device)
