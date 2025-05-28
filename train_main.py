import torch
import torch.nn as nn
import torch.optim as optim
from models.transformer import TransformerClassifier
from utils.preprocessing import build_dataloaders, build_char_to_idx

def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_accuracy = correct / total
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}, Accuracy: {train_accuracy:.4f}")

        evaluate_model(model, val_loader, criterion, device)

def evaluate_model(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_accuracy = correct / total
    print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Transformer model for malicious URL detection")
    parser.add_argument("--data_path", type=str, default="malicious_url_detector/data/malicious_phish_CSV.csv", help="Path to the CSV data file")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--max_len", type=int, default=200, help="Maximum length of URL sequences")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of output classes")
    parser.add_argument("--d_model", type=int, default=128, help="Dimension of model embeddings")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of Transformer encoder layers")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=512, help="Dimension of feedforward network")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--max_len_pos", type=int, default=200, help="Maximum length for positional encoding")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader, vocab_size = build_dataloaders(
        args.data_path,
        batch_size=args.batch_size,
        max_len=args.max_len,
        num_classes=args.num_classes
    )

    model = TransformerClassifier(
        vocab_size=vocab_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        num_classes=args.num_classes,
        dropout=args.dropout,
        max_len=args.max_len_pos
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print("Starting training...")
    train_model(model, train_loader, val_loader, criterion, optimizer, device, args.epochs)

    print("Evaluating on test set...")
    evaluate_model(model, test_loader, criterion, device)

    torch.save({
        'model_state_dict': model.state_dict(),
        'char2idx': build_char_to_idx()
    }, "malicious_url_detector/saved_model/transformer_model.pth")
    print("Model saved to malicious_url_detector/saved_model/transformer_model.pth")
