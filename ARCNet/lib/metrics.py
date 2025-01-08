import torch
from sklearn import metrics
import numpy as np


def compute_accuracy(y_pred, y_true):
    if torch.is_tensor(y_pred):
        y_pred = y_pred.data.cpu().numpy()
    if torch.is_tensor(y_true):
        y_true = y_true.data.cpu().numpy()
    if y_pred.shape[1] == 1:
        y_pred = y_pred[:, 0]
    else:
        y_pred = np.argmax(y_pred, axis=1)
    return metrics.accuracy_score(y_true, y_pred, normalize=True)


def compute_f1(y_pred, y_true):
    if torch.is_tensor(y_pred):
        y_pred = y_pred.data.cpu().numpy()
    if torch.is_tensor(y_true):
        y_true = y_true.data.cpu().numpy()
    if y_pred.shape[1] == 1:
        y_pred = y_pred[:, 0]
    else:
        y_pred = np.argmax(y_pred, axis=1)
    # 'micro', 'macro', 'weighted'
    return metrics.f1_score(y_true, y_pred, average='weighted')

def compute_average_accuracy(y_pred, y_true):
    if torch.is_tensor(y_pred):
        y_pred = y_pred.data.cpu().numpy()
    if torch.is_tensor(y_true):
        y_true = y_true.data.cpu().numpy()
    if y_pred.shape[1] == 1:
        y_pred = y_pred[:, 0]
    else:
        y_pred = np.argmax(y_pred, axis=1)

    unique_labels = np.unique(y_true)
    average_accuracies = {}

    for label in unique_labels:
        label_indices = np.where(y_true == label)[0]
        if len(label_indices) > 0:
            label_pred = y_pred[label_indices]
            label_accuracy = np.mean(label_pred == label)
            average_accuracies[label] = label_accuracy
    sorted_average_accuracies = {k: v for k, v in sorted(average_accuracies.items(), key=lambda item: item[0])}
    return sorted_average_accuracies


def compute_confusion_matrix(y_pred, y_true, num_classes=45):
    if torch.is_tensor(y_pred):
        y_pred = y_pred.data.cpu().numpy()
    if torch.is_tensor(y_true):
        y_true = y_true.data.cpu().numpy()
    
    if y_pred.ndim > 1 and y_pred.shape[1] == 1:
        y_pred = y_pred[:, 0]
    elif y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=1)
    
    cm = metrics.confusion_matrix(y_true, y_pred, labels=range(num_classes))
    return cm