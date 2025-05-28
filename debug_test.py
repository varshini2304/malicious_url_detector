import torch
from predict import load_model, predict_url

def test_url_prediction():
    url = "http://gOogle.com"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, char2idx = load_model("malicious_url_detector/saved_model/transformer_model.pth", device)
    pred_class = predict_url(model, char2idx, url, device=device)
    if pred_class == 1:
        print(f"The URL '{url}' is recognized as MALICIOUS.")
    else:
        print(f"The URL '{url}' is recognized as BENIGN.")

if __name__ == "__main__":
    test_url_prediction()
