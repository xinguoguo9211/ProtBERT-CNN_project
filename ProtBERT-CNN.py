"""
ProtBERT-CNN
============

Frozen ProtBERT embedding layer coupled with a compact one-dimensional
convolutional classifier for peptide-TCR (CDR3beta) binding prediction.

The ProtBERT component is never updated. Only the downstream CNN is trained.

Input data are CSV files with at least three columns.
    peptide      antigen peptide sequence
    binding_TCR  CDR3beta amino acid sequence
    label        1 for binding, 0 for non-binding

Usage examples
--------------
Train and evaluate on the internal balanced sets.

    python ProtBERT-CNN.py \
        --train_csv data/majority_model_dataset.csv \
        --test_csv data/majority_test_dataset.csv \
        --output_dir output

Run positive-only recognition on an independent external cohort with a
previously saved checkpoint.

    python ProtBERT-CNN.py \
        --checkpoint output/prot_bert_model.pth \
        --external_csv data/traitdb_positive.csv \
        --output_dir output
"""

import argparse
import logging
import os
import random
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import Dataset
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertModel, BertTokenizer

warnings.filterwarnings("ignore")
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_IB_DISABLE", "1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")


# ---------------------------------------------------------------------------
# Reproducibility and data loading
# ---------------------------------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_csv_safe(path):
    """Read a CSV file with fallback encodings for legacy GBK exports."""
    for encoding in ("utf-8", "gbk", "latin-1"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("Unable to read {} with supported encodings.".format(path))


def create_dataset(tokenizer, df, max_length=512, keep_columns=False):
    """Tokenize peptide and CDR3beta pairs and build a Hugging Face Dataset."""
    seqs_df = df.copy()
    seqs_df["peptide"] = seqs_df["peptide"].astype(str).str.replace(
        "|".join(["O", "B", "U", "Z"]), "X", regex=True
    )
    seqs_df["binding_TCR"] = seqs_df["binding_TCR"].astype(str).str.replace(
        "|".join(["O", "B", "U", "Z"]), "X", regex=True
    )
    seqs_df["peptide"] = seqs_df.apply(
        lambda row: " ".join(row["peptide"]), axis=1
    )
    seqs_df["binding_TCR"] = seqs_df.apply(
        lambda row: " ".join(row["binding_TCR"]), axis=1
    )

    tokenized = tokenizer(
        list(seqs_df["peptide"]),
        list(seqs_df["binding_TCR"]),
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    dataset = Dataset.from_dict(tokenized)
    if keep_columns:
        dataset = dataset.add_column("peptide", list(seqs_df["peptide"]))
        dataset = dataset.add_column("binding_TCR", list(seqs_df["binding_TCR"]))
    dataset = dataset.add_column("labels", list(seqs_df["label"]))
    dataset = dataset.with_format("torch")
    return dataset


# ---------------------------------------------------------------------------
# Frozen ProtBERT embeddings
# ---------------------------------------------------------------------------
def get_batch_prot_bert_embeddings(batch, model, device):
    """Return residue-level ProtBERT embeddings without CLS and SEP tokens."""
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    embeddings = outputs.last_hidden_state
    batch_size = embeddings.size(0)

    # Exclude the final SEP token from the pooled residue representation.
    for k in range(batch_size):
        last_one_index = (
            (attention_mask[k] == 1).nonzero(as_tuple=True)[0][-1].item()
        )
        attention_mask[k, last_one_index] = 0

    mask = attention_mask.unsqueeze(-1)
    embeddings = embeddings * mask

    del input_ids, attention_mask, outputs, mask
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ProtBERT adds one special token at each end. Remove both.
    return embeddings[:, 1:-1, :]


# ---------------------------------------------------------------------------
# CNN classifier
# ---------------------------------------------------------------------------
class CNN1d(nn.Module):
    def __init__(self, first_channels_num):
        super(CNN1d, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=first_channels_num,
                out_channels=first_channels_num * 2,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, ceil_mode=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=first_channels_num * 2,
                out_channels=first_channels_num * 4,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, ceil_mode=True),
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(first_channels_num * 4, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ---------------------------------------------------------------------------
# Training and validation
# ---------------------------------------------------------------------------
def validate_cnn(model, valid_loader, prot_bert, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(valid_loader, desc="Validating", dynamic_ncols=True):
            inputs = get_batch_prot_bert_embeddings(batch, prot_bert, device)
            labels = batch["labels"].to(device)

            outputs = model(inputs.permute(0, 2, 1))
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return val_loss / len(valid_loader), correct / total


def train_cnn(
    model,
    train_loader,
    valid_loader,
    prot_bert,
    criterion,
    optimizer,
    scheduler,
    device,
    num_epochs,
    save_path,
    log_path,
):
    best_val_loss = float("inf")

    with open(log_path, "w", encoding="utf-8") as log_file:
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            correct = 0
            total = 0

            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            print("Epoch [{}/{}]".format(epoch + 1, num_epochs))

            for batch in tqdm(train_loader, desc="Training", dynamic_ncols=True):
                inputs = get_batch_prot_bert_embeddings(
                    batch, prot_bert, device
                )
                labels = batch["labels"].to(device)

                optimizer.zero_grad()
                outputs = model(inputs.permute(0, 2, 1))
                loss = criterion(outputs, labels)
                train_loss += loss.item()

                loss.backward()
                optimizer.step()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            avg_train_loss = train_loss / len(train_loader)
            train_accuracy = correct / total

            val_loss, val_accuracy = validate_cnn(
                model, valid_loader, prot_bert, criterion, device
            )

            print(
                "Train Loss: {:.4f}, Train Accuracy: {:.4f}".format(
                    avg_train_loss, train_accuracy
                )
            )
            print(
                "Validation Loss: {:.4f}, Validation Accuracy: {:.4f}".format(
                    val_loss, val_accuracy
                )
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), save_path)
                print(
                    "Saved best model with validation loss {:.4f}".format(
                        val_loss
                    )
                )

            log_file.write(
                time.strftime("%Y-%m-%d %H:%M:%S\n", time.localtime())
            )
            log_file.write(
                "Epoch [{}/{}]\n".format(epoch + 1, num_epochs)
            )
            log_file.write(
                "Train Loss: {:.4f}, Train Accuracy: {:.4f}\n".format(
                    avg_train_loss, train_accuracy
                )
            )
            log_file.write(
                "Validation Loss: {:.4f}, Validation Accuracy: {:.4f}\n".format(
                    val_loss, val_accuracy
                )
            )
            log_file.flush()

            scheduler.step()


# ---------------------------------------------------------------------------
# Internal balanced-set evaluation
# ---------------------------------------------------------------------------
def metrics_cal(predictions):
    y_true = predictions["true_label"]
    y_pred = predictions["pred_label"]
    proba = predictions["proba_1"]

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = recall_score(y_true, y_pred)
    specificity = tn / (tn + fp)
    accuracy = (tn + tp) / (tn + fp + fn + tp)
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    auc = roc_auc_score(y_true, proba)

    print("Sensitivity (SN): {:.4f}".format(sensitivity))
    print("Specificity (SP): {:.4f}".format(specificity))
    print("ACC: {:.4f}".format(accuracy))
    print("F1-score: {:.4f}".format(f1))
    print("MCC: {:.4f}".format(mcc))
    print("AUC: {:.4f}".format(auc))

    return sensitivity, specificity, accuracy, f1, mcc, auc


def test_cnn(model, test_loader, prot_bert, device):
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing", dynamic_ncols=True):
            inputs = get_batch_prot_bert_embeddings(batch, prot_bert, device)
            true_label = batch["labels"].to(device)

            outputs = model(inputs.permute(0, 2, 1))
            proba = torch.softmax(outputs, dim=1)
            pred_label = torch.argmax(proba, dim=1)

            for i in range(proba.size(0)):
                predictions.append(
                    {
                        "proba_0": proba[i, 0].item(),
                        "proba_1": proba[i, 1].item(),
                        "pred_label": pred_label[i].item(),
                        "true_label": true_label[i].item(),
                    }
                )

    predictions = pd.DataFrame(predictions)
    metrics = metrics_cal(predictions)
    return predictions, metrics


# ---------------------------------------------------------------------------
# Positive-only external recognition
# ---------------------------------------------------------------------------
def recognize_positive(model, external_loader, prot_bert, device):
    model.eval()
    rows = []
    total = 0
    recognized = 0

    with torch.no_grad():
        for batch in tqdm(
            external_loader, desc="Recognizing", dynamic_ncols=True
        ):
            inputs = get_batch_prot_bert_embeddings(batch, prot_bert, device)
            outputs = model(inputs.permute(0, 2, 1))

            proba = torch.softmax(outputs, dim=1)
            pred_label = torch.argmax(proba, dim=1)

            batch_size = proba.size(0)
            total += batch_size
            recognized += pred_label.sum().item()

            for i in range(batch_size):
                rows.append(
                    {
                        "proba_0": proba[i, 0].item(),
                        "proba_1": proba[i, 1].item(),
                        "pred_label": pred_label[i].item(),
                    }
                )

    recognition_rate = recognized / total
    print(
        "Recognized {}/{} pairs, recognition rate {:.4f}".format(
            recognized, total, recognition_rate
        )
    )
    return pd.DataFrame(rows), recognized, total, recognition_rate


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="ProtBERT-CNN peptide-TCR binding prediction"
    )
    parser.add_argument(
        "--train_csv",
        type=str,
        default=None,
        help="CSV file for model building, split into train and validation.",
    )
    parser.add_argument(
        "--test_csv",
        type=str,
        default=None,
        help="Balanced internal test CSV with peptide, binding_TCR and label.",
    )
    parser.add_argument(
        "--external_csv",
        type=str,
        default=None,
        help="Positive-only external CSV for recognition rate evaluation.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Existing CNN checkpoint. Required when train_csv is not given.",
    )
    parser.add_argument(
        "--plm_path",
        type=str,
        default="Rostlab/prot_bert",
        help="Local ProtBERT directory or Hugging Face model identifier.",
    )
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--test_batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--valid_size", type=float, default=0.2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    pin_memory = device.type == "cuda"
    os.makedirs(args.output_dir, exist_ok=True)

    save_path = os.path.join(args.output_dir, "prot_bert_model.pth")
    log_path = os.path.join(args.output_dir, "prot_bert_model_history.txt")

    tokenizer = BertTokenizer.from_pretrained(args.plm_path)
    prot_bert = BertModel.from_pretrained(args.plm_path).to(device)
    prot_bert.eval()
    for param in prot_bert.parameters():
        param.requires_grad = False

    cnn = CNN1d(first_channels_num=1024).to(device)

    if args.train_csv is not None:
        from sklearn.model_selection import train_test_split

        data_all = read_csv_safe(args.train_csv)
        train_df, valid_df = train_test_split(
            data_all,
            test_size=args.valid_size,
            random_state=args.seed,
        )

        train_data = create_dataset(
            tokenizer, train_df, max_length=args.max_length
        )
        valid_data = create_dataset(
            tokenizer, valid_df, max_length=args.max_length
        )

        train_loader = DataLoader(
            train_data,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )
        valid_loader = DataLoader(
            valid_data,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(cnn.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=args.gamma
        )

        train_cnn(
            model=cnn,
            train_loader=train_loader,
            valid_loader=valid_loader,
            prot_bert=prot_bert,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            num_epochs=args.epochs,
            save_path=save_path,
            log_path=log_path,
        )
        args.checkpoint = save_path

    if args.checkpoint is None:
        raise ValueError(
            "Provide --train_csv for training or --checkpoint for evaluation."
        )

    cnn.load_state_dict(torch.load(args.checkpoint, map_location=device))
    cnn.to(device)

    if args.test_csv is not None:
        test_df = read_csv_safe(args.test_csv)
        test_data = create_dataset(
            tokenizer,
            test_df,
            max_length=args.max_length,
            keep_columns=True,
        )
        test_loader = DataLoader(
            test_data,
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )
        predictions, metrics = test_cnn(cnn, test_loader, prot_bert, device)
        predictions.to_csv(
            os.path.join(args.output_dir, "test_predictions.csv"),
            index=False,
        )
        metric_names = [
            "sensitivity",
            "specificity",
            "accuracy",
            "f1",
            "mcc",
            "auc",
        ]
        pd.DataFrame(
            {"metric": metric_names, "value": metrics}
        ).to_csv(
            os.path.join(args.output_dir, "test_metrics.csv"), index=False
        )

    if args.external_csv is not None:
        external_df = read_csv_safe(args.external_csv)
        if "label" not in external_df.columns:
            external_df["label"] = 1

        external_data = create_dataset(
            tokenizer,
            external_df,
            max_length=args.max_length,
            keep_columns=True,
        )
        external_loader = DataLoader(
            external_data,
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )
        external_pred, recognized, total, rate = recognize_positive(
            cnn, external_loader, prot_bert, device
        )
        external_pred.to_csv(
            os.path.join(args.output_dir, "external_predictions.csv"),
            index=False,
        )
        with open(
            os.path.join(args.output_dir, "external_recognition.txt"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(
                "Recognized {}/{} pairs, recognition rate {:.4f}\n".format(
                    recognized, total, rate
                )
            )


if __name__ == "__main__":
    main()
#（注：内容由AI生成）
