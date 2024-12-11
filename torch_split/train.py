

import torch
from preprocessing import AttentionModule,DescriptorGraphDataset
import torch.nn as nn
import numpy as np
import os
from torch_geometric.loader import DataLoader as GeoDataLoader
from model import GATClassifier
import pandas as pd
from tqdm.auto import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    total_acc = 0
    for batch in tqdm(loader, desc="Training"):
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(
            batch.x, batch.edge_index, batch.batch, edge_attr=batch.edge_attr
        ).squeeze()
        loss = criterion(logits, batch.y)
        loss.backward()
        optimizer.step()
        preds = torch.sigmoid(logits)
        acc = ((preds > 0.5).float() == batch.y).float().mean()
        total_loss += loss.item()
        total_acc += acc.item()
    return total_loss / len(loader), total_acc / len(loader)


def validate(model, loader, criterion):
    model.eval()
    total_loss = 0
    total_acc = 0
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating"):
            batch = batch.to(device)
            logits = model(
                batch.x, batch.edge_index, batch.batch, edge_attr=batch.edge_attr
            ).squeeze()
            loss = criterion(logits, batch.y)
            preds = torch.sigmoid(logits)
            acc = ((preds > 0.5).float() == batch.y).float().mean()
            total_loss += loss.item()
            total_acc += acc.item()
            all_labels.extend(batch.y.cpu().numpy())
            all_probs.extend(preds.cpu().numpy())
    avg_loss = total_loss / len(loader)
    avg_acc = total_acc / len(loader)
    return avg_loss, avg_acc, all_labels, all_probs


if __name__ == "__main__":
    dataset_dir = "ADM_dataset"
    exp_name = "ADM"

    os.makedirs("models", exist_ok=True)

    # 初始化注意力模塊
    attention_module = AttentionModule(input_dim=256, dk=64).to(device)
    attention_module.eval()  

    train_set = DescriptorGraphDataset(
        os.path.join(dataset_dir, "train"),
        mode="train",
        attention_module=attention_module,
    )
    valid_set = DescriptorGraphDataset(
        os.path.join(dataset_dir, "valid"),
        mode="valid",
        attention_module=attention_module,
    )

    # DataLoader
    batch_size = 64  # 減少批次大小
    train_loader = GeoDataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=0
    )
    valid_loader = GeoDataLoader(
        valid_set, batch_size=batch_size, shuffle=False, num_workers=0
    )

    sample_data = None
    for data in train_set:
        if data is not None:
            sample_data = data
            break
    if sample_data is None:
        print("No valid data found in training set.")
        exit()
    input_dim = sample_data.num_node_features

    model = GATClassifier(input_dim).to(device)
    # Binary classification 使用 BCEWithLogitsLoss
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    n_epochs = 50  
    patience = 10
    best_acc = 0
    stale = 0
    train_losses, valid_losses, train_accuracies, valid_accuracies = [], [], [], []

    for epoch in range(n_epochs):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion)
        valid_loss, valid_acc, valid_labels, valid_probs = validate(
            model, valid_loader, criterion
        )

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        train_accuracies.append(train_acc)
        valid_accuracies.append(valid_acc)

        print(
            f"[Epoch {epoch + 1:03d}/{n_epochs:03d}] "
            f"Train Loss: {train_loss:.5f}, Train Acc: {train_acc:.5f} | "
            f"Valid Loss: {valid_loss:.5f}, Valid Acc: {valid_acc:.5f}"
        )

        scheduler.step()

        if valid_acc > best_acc:
            print(f"Best model found at epoch {epoch + 1}, saving model")
            torch.save(model.state_dict(), f"models/{exp_name}_best.ckpt")
            best_acc = valid_acc
            stale = 0
            # 用來繪製 ROC
            best_valid_labels = valid_labels
            best_valid_probs = valid_probs
        else:
            stale += 1
            if stale > patience:
                print(
                    f"No improvement in {patience} consecutive epochs, early stopping"
                )
                break

    test_set = DescriptorGraphDataset(
        os.path.join(dataset_dir, "test"),
        mode="test",
        attention_module=attention_module,
    )
    test_loader = GeoDataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=4
    )

    model_file = f"models/{exp_name}_best.ckpt"
    model_best = GATClassifier(input_dim).to(device)
    model_best.load_state_dict(
        torch.load(model_file, map_location=device)
    )  # 使用 map_location
    model_best.eval()

    prediction = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            batch = batch.to(device)
            logits = model_best(
                batch.x, batch.edge_index, batch.batch, edge_attr=batch.edge_attr
            ).squeeze()
            preds = torch.sigmoid(logits)
            prediction.extend((preds > 0.5).long().cpu().numpy())

    df = pd.DataFrame()
    df["Id"] = [str(i).zfill(4) for i in range(1, len(test_set) + 1)]
    df["Category"] = prediction
    df.to_csv("submission.csv", index=False)
    print("Prediction results saved to submission.csv")

    if "best_valid_labels" in locals() and "best_valid_probs" in locals():
        best_valid_labels = np.array(best_valid_labels)
        best_valid_probs = np.array(best_valid_probs)

        fpr, tpr, thresholds = roc_curve(best_valid_labels, best_valid_probs)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        lw = 2
        plt.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=lw,
            label="ROC curve (AUC = %0.4f)" % roc_auc,
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC)")
        plt.legend(loc="lower right")
        plt.show()
    else:
        print("No validation data available to compute ROC curve.")
