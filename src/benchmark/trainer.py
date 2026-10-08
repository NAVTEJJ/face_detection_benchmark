import time
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.config import NUM_EPOCHS_CNN, NUM_EPOCHS_QNN, LEARNING_RATE, BATCH_SIZE


def train_cnn(model, train_loader, device="cpu"):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    history = {"loss": [], "acc": []}
    t0 = time.time()

    for epoch in range(NUM_EPOCHS_CNN):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(y_batch)
            correct      += (logits.argmax(dim=1) == y_batch).sum().item()
            total        += len(y_batch)

        epoch_loss = running_loss / total
        epoch_acc  = correct / total
        history["loss"].append(epoch_loss)
        history["acc"].append(epoch_acc)
        print(f"  CNN Epoch {epoch+1:02d}/{NUM_EPOCHS_CNN}  loss={epoch_loss:.4f}  acc={epoch_acc:.4f}")

    train_time = time.time() - t0
    print(f"  CNN training done in {train_time:.1f}s")
    return history, train_time


def train_qnn(model, feat_train, y_train_tensor, device="cpu"):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    ds     = TensorDataset(feat_train, y_train_tensor)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    history = {"loss": [], "acc": []}
    t0 = time.time()

    for epoch in range(NUM_EPOCHS_QNN):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for feat_batch, y_batch in loader:
            feat_batch, y_batch = feat_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model.forward_from_features(feat_batch)
            loss   = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(y_batch)
            correct      += (logits.argmax(dim=1) == y_batch).sum().item()
            total        += len(y_batch)

        epoch_loss = running_loss / total
        epoch_acc  = correct / total
        history["loss"].append(epoch_loss)
        history["acc"].append(epoch_acc)
        print(f"  QNN Epoch {epoch+1:02d}/{NUM_EPOCHS_QNN}  loss={epoch_loss:.4f}  acc={epoch_acc:.4f}")

    train_time = time.time() - t0
    print(f"  QNN training done in {train_time:.1f}s")
    return history, train_time
