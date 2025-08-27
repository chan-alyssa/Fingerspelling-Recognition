import torch
import torch.nn as nn
import torch.optim as optim
import h5py
import copy
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib
import cv2
from torch.utils.data import Dataset, DataLoader, TensorDataset
from imblearn.over_sampling import RandomOverSampler
import os
import time


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BinaryClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=128, dropout_prob=0.2):
        super(BinaryClassifier, self).__init__()
        
        # First hidden layer
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.layer_norm1 = nn.LayerNorm(hidden_size)  
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_prob)
        
        # Second hidden layer 
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.layer_norm2 = nn.LayerNorm(hidden_size // 2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_prob)

        # Third hidden layer
        self.fc3 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.layer_norm3 = nn.LayerNorm(hidden_size // 4)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout_prob)
        
        # Output layer
        self.fc4 = nn.Linear(hidden_size // 4, 1)  

    def forward(self, x):
        x = self.fc1(x)
        x = self.layer_norm1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.layer_norm2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        x = self.layer_norm3(x)
        x = self.relu3(x)
        x = self.dropout3(x)

        x = self.fc4(x) 
        return x


def calculate_f1_score(TP, FP, FN):
    """Calculate F1-score from TP, FP, FN."""
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1


def plot_metrics(train_losses, train_losses_oversample, test_losses, train_accuracies, test_accuracies, train_f1_scores, test_f1_scores,
                 train_class_0_accuracies, train_class_1_accuracies, test_class_0_accuracies, test_class_1_accuracies,
                 train_TP, train_TN, train_FP, train_FN, test_TP, test_TN, test_FP, test_FN):
    plt.figure(figsize=(12, 8))
    plt.plot(train_losses_oversample, label='Train Loss Oversampled', linestyle='--', linewidth=2)
    plt.plot(test_losses, label='Test Loss', linestyle=':', linewidth=2)
    plt.title('Loss Curve', fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.savefig('132loss_curve.png', bbox_inches='tight', dpi=300)

    plt.figure(figsize=(12, 8))
    plt.plot(test_accuracies, label='Test Average Accuracy', linestyle='-', linewidth=2)
    plt.plot(train_accuracies, label='Train Average Accuracy', linestyle='--', linewidth=2)
    plt.title('Average Accuracy Curve', fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.legend(fontsize=12)
    plt.savefig('average_accuracy_curve.png', bbox_inches='tight', dpi=300)

    plt.figure(figsize=(12, 8))
    plt.plot(train_TP, label='Train TP', linestyle='-', linewidth=2)
    plt.plot(train_TN, label='Train TN', linestyle='--', linewidth=2)
    plt.plot(train_FP, label='Train FP', linestyle=':', linewidth=2)
    plt.plot(train_FN, label='Train FN', linestyle='-.', linewidth=2)
    plt.title('Train TP, TN, FP, FN', fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.legend(fontsize=12)
    plt.savefig('train_tp_tn_fp_fn_curve.png', bbox_inches='tight', dpi=300)

    plt.figure(figsize=(12, 8))
    plt.plot(test_TP, label='Test TP', linestyle='-', linewidth=2)
    plt.plot(test_TN, label='Test TN', linestyle='--', linewidth=2)
    plt.plot(test_FP, label='Test FP', linestyle=':', linewidth=2)
    plt.plot(test_FN, label='Test FN', linestyle='-.', linewidth=2)
    plt.title('Test TP, TN, FP, FN', fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.legend(fontsize=12)
    plt.savefig('test_tp_tn_fp_fn_curve.png', bbox_inches='tight', dpi=300)
    
    plt.figure(figsize=(12, 8))
    plt.plot(train_class_0_accuracies, label='Train Class 0 Accuracy', linestyle='-', linewidth=2)
    plt.plot(train_class_1_accuracies, label='Train Class 1 Accuracy', linestyle='--', linewidth=2)
    plt.plot(test_class_0_accuracies, label='Test Class 0 Accuracy', linestyle=':', linewidth=2)
    plt.plot(test_class_1_accuracies, label='Test Class 1 Accuracy', linestyle='-.', linewidth=2)
    plt.title('Class-wise Accuracy Curve', fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.legend(fontsize=12)
    plt.savefig('class_wise_accuracy_curve.png', bbox_inches='tight', dpi=300)

def create_video_indices_dict(video_names):
    # Using defaultdict to avoid key errors and streamline the process
    from collections import defaultdict
    video_indices_dict = defaultdict(list)
    for idx, video in enumerate(video_names):
        video_indices_dict[int(video.item())].append(idx)
    return video_indices_dict

def fingerspelling(predictions, window_size, threshold):
    step_size=1
    num_frames = len(predictions)

    fingerspelling_values = np.zeros(num_frames)

    for start in range(0, num_frames - window_size + 1, step_size):
        end = start + window_size
        window_predictions = predictions[start:end]
        proportion_active = torch.sum(window_predictions)
        fingerspelling = 1 if proportion_active > threshold else 0

        if fingerspelling == 1:
            fingerspelling_values[start:end] = 1

    return fingerspelling_values


def find_events(binary_array):

    events = []
    in_event = False
    start = None

    for i, val in enumerate(binary_array):
        if val == 1 and not in_event:
            in_event = True
            start = i
        elif val == 0 and in_event:
            in_event = False
            events.append((start, i - 1))
    if in_event:
        events.append((start, len(binary_array) - 1))
    return events

def calculate_iou_and_metrics(predictions, ground_truth, video_list, iou_threshold=0.5):
    tp = 0
    fp = 0
    fn = 0

    ground_event = False  
    pred_event = False 

    intersection = 0
    union = 0
    total_gt_events = 0

    previous_video = None 

    for i in range(len(predictions)):
        pred = predictions[i]
        gt = ground_truth[i]
        video_name = video_list[i]

        if video_name != previous_video:
            intersection = 0
            union = 0
            ground_event = False
            pred_event = False

        previous_video = video_name 

        if gt == 1 and not ground_event:
            ground_event = True
            total_gt_events += 1 

        if pred == 1 and not pred_event:
            pred_event = True

        if ground_event or pred_event:
            union += 1
        if ground_event and pred_event:
            intersection += 1

        if pred == 0 and pred_event and not ground_event:
            if intersection==0:
                fp += 1 
                pred_event = False
                intersection = 0
                union = 0

        if (pred == 0 and gt == 0) and (pred_event or ground_event):
            iou = intersection / union if union > 0 else 0
            if iou >= iou_threshold:
                tp += 1  
            else:
                fn += 1  
            pred_event = False
            ground_event = False
            intersection = 0
            union = 0

        if gt == 0 and ground_event:
            ground_event = False  
        
        if pred == 0 and pred_event:
            pred_event = False 

    fn = total_gt_events - tp 
 
    return tp, fn, fp

def oversample_random(X, y):

    y = np.squeeze(y)
    X_majority = X[y == 0]
    X_minority = X[y == 1]

    minority_oversampled = np.random.choice(len(X_minority), size=len(X_majority), replace=True)
    X_minority_oversampled = X_minority[minority_oversampled]

    X_balanced = np.vstack([X_majority, X_minority_oversampled])
    y_balanced = np.hstack([np.zeros(len(X_majority)), np.ones(len(X_minority_oversampled))])

    indices = np.arange(len(y_balanced))
    np.random.shuffle(indices)
    return X_balanced[indices], y_balanced[indices]


def train_and_evaluate(model, X_train_oversample, y_train_oversample,X_test, y_test, video_list,learning_rate, num_epochs):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_epoch = -1
    best_accuracy = 0.0
    best_model = None
    best_preds = None  

    train_losses, train_losses_oversample, test_losses = [], [],[]
    train_accuracies, test_accuracies = [], []
    train_f1_scores, test_f1_scores = [], []
    train_class_0_accuracies, train_class_1_accuracies = [], []
    test_class_0_accuracies, test_class_1_accuracies = [], []
    train_TP, train_TN, train_FP, train_FN = [], [], [], []
    test_TP, test_TN, test_FP, test_FN = [], [], [], []

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        outputs_oversample = model(X_train_oversample)

        loss_oversample = criterion(outputs_oversample, y_train_oversample)

        loss_oversample.backward()
        optimizer.step() 

        train_losses_oversample.append(loss_oversample.item())

        train_preds = (torch.sigmoid(outputs_oversample) > 0.5).float()
        TP = ((train_preds == 1) & (y_train_oversample == 1)).sum().item()
        FN = ((train_preds == 0) & (y_train_oversample == 1)).sum().item()
        FP = ((train_preds == 1) & (y_train_oversample == 0)).sum().item()
        TN = ((train_preds == 0) & (y_train_oversample == 0)).sum().item()
        train_TP.append(TP)
        train_FN.append(FN)
        train_FP.append(FP)
        train_TN.append(TN)

        train_f1_scores.append(calculate_f1_score(TP, FP, FN))
        trainclass0= TN /((y_train_oversample == 0).sum().item() )
        train_class_0_accuracies.append(trainclass0)
        trainclass1= TP / (y_train_oversample == 1).sum().item()
        train_class_1_accuracies.append(trainclass1)
        train_accuracy = (trainclass1+trainclass0)/2
        train_accuracies.append(train_accuracy)

        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, y_test)
            test_losses.append(test_loss.item())
            test_preds = (torch.sigmoid(test_outputs) > 0.5).float()

            TP_test = ((test_preds == 1) & (y_test == 1)).sum().item()
            FN_test = ((test_preds == 0) & (y_test == 1)).sum().item()
            FP_test = ((test_preds == 1) & (y_test == 0)).sum().item()
            TN_test = ((test_preds == 0) & (y_test == 0)).sum().item()

            test_TP.append(TP_test)
            test_FN.append(FN_test)
            test_FP.append(FP_test)
            test_TN.append(TN_test)

            test_f1_scores.append(calculate_f1_score(TP_test, FP_test, FN_test))
            class0= TN_test /((y_test == 0).sum().item() )

            test_class_0_accuracies.append(class0)
            class1= TP_test / (y_test == 1).sum().item()
            test_class_1_accuracies.append(class1)
            test_accuracy = (class1+class0)/2

            test_accuracies.append(test_accuracy)
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                best_epoch = epoch
                best_model = copy.deepcopy(model)
                best_preds = test_preds.detach().clone()
                best_model = model 
                best_preds = test_preds.detach().clone()
                torch.save(model.state_dict(), 'binary_classifierf86.pth')

        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{num_epochs}, Train Loss: {loss_oversample:.4f}, Test Loss: {test_loss:.4f}, Train Acc: {train_accuracy:.4f}, Test Acc: {test_accuracy:.4f}")
    best_model = model  # Reuse the model architecture
    best_model.load_state_dict(torch.load('binary_classifierf86.pth'))  
    results = fingerspelling(best_preds, 5,3)

    tp ,fn, fp = calculate_iou_and_metrics(results, y_test,video_list)
    print(tp,fn,fp)
    print(TP_test,FN_test,FP_test)
    print(best_accuracy)

    return train_losses, train_losses_oversample, test_losses, train_accuracies, test_accuracies, train_f1_scores, test_f1_scores, train_class_0_accuracies, train_class_1_accuracies, test_class_0_accuracies, test_class_1_accuracies, train_TP, train_TN, train_FP, train_FN, test_TP, test_TN, test_FP, test_FN

class BOBSLDataset(Dataset):
    def __init__(self, file_path, transform=None, target_transform=None):
        with h5py.File(file_path, 'r') as f:
            self.feature = f['video/features'][:]
            self.label = f['video/label'][:]
            self.frame_number = f['video/frame_number'][:]
            self.video_list = f['video/video_name'][:]

    def __len__(self):
        return len(self.feature)

    def __getitem__(self, idx):
        feature = self.feature[idx]
        label = self.label[idx] 
        return feature, label

def main():
    parser = argparse.ArgumentParser(description="Binary Classifier Training for BOBSL Dataset")
    parser.add_argument('--train-file', type=str, default='training1f86.h5', help='Path to training H5 file')
    parser.add_argument('--test-file', type=str, default='testing1f86.h5', help='Path to test H5 file')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--oversample', action='store_true', help='Use random oversampling for balancing classes')
    args = parser.parse_args()

    training_data = BOBSLDataset(train_file)
    testing_data = BOBSLDataset(test_file)

    X_train = training_data.feature
    y_train = training_data.label

    X_test = testing_data.feature
    y_test = testing_data.label
    video_list = testing_data.video_list
    if args.oversample:
        ros = RandomOverSampler(sampling_strategy='auto')
        X_train_oversample, y_train_oversample = ros.fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_train_oversample = scaler.fit_transform(X_train_oversample)
    X_test = scaler.transform(X_test)

    joblib.dump(scaler, 'scalerf86.pkl')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train_tensor = torch.tensor(X_train_oversample, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train_oversample, dtype=torch.float32).to(device).unsqueeze(1)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

    model = BinaryClassifier(X_train_tensor.shape[1]).to(device)

    results = train_and_evaluate(
        model, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, video_list,
        learning_rate=args.learning_rate, num_epochs=args.epochs
    )
    plot_metrics(*results)


if __name__ == "__main__":
    main()
