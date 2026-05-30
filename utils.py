import os
import torch
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# -----------------------------
# 1. GET CLASS NAMES
# -----------------------------
def get_classes(train_dir):
    """
    Reads class names from dataset folder.
    Example:
    dataset/train/RBC, WBC, Platelets
    """
    classes = sorted([
        d for d in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, d))
    ])
    return classes


# -----------------------------
# 2. SAVE MODEL
# -----------------------------
def save_model(model, path="model.pth"):
    """
    Saves PyTorch model weights safely.
    """
    torch.save(model.state_dict(), path)
    print(f"✅ Model saved at: {path}")


# -----------------------------
# 3. LOAD MODEL
# -----------------------------
def load_model(model, path, device="cpu"):
    """
    Loads PyTorch model weights safely.
    """
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    print(f"✅ Model loaded from: {path}")
    return model


# -----------------------------
# 4. ACCURACY CALCULATION
# -----------------------------
def calculate_accuracy(y_true, y_pred):
    """
    Returns accuracy score.
    """
    return accuracy_score(y_true, y_pred)


# -----------------------------
# 5. CONFUSION MATRIX
# -----------------------------
def get_confusion_matrix(y_true, y_pred, labels=None):
    """
    Returns confusion matrix.
    """
    return confusion_matrix(y_true, y_pred, labels=labels)


# -----------------------------
# 6. CLASSIFICATION REPORT
# -----------------------------
def get_classification_report(y_true, y_pred, target_names=None):
    """
    Precision, Recall, F1-score report.
    """
    return classification_report(
        y_true,
        y_pred,
        target_names=target_names
    )


# -----------------------------
# 7. PREDICT SINGLE IMAGE
# -----------------------------
def predict_image(model, image_tensor, class_names):
    """
    Returns predicted class + confidence.
    """
    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.softmax(output, dim=1)
        conf, pred = torch.max(probs, 1)

    return class_names[pred.item()], conf.item()


# -----------------------------
# 8. CONVERT LISTS TO NUMPY (safe utility)
# -----------------------------
def to_numpy(x):
    """
    Converts tensor/list to numpy array safely.
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.array(x)