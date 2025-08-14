import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import h5py
from torch.utils.data import Dataset, DataLoader

# Custom Dataset for loading .h5 files

class BOBSLDataset(Dataset):
    def __init__(self, file_path, transform=None, target_transform=None):
        with h5py.File(file_path, 'r') as f:
            self.feature = f['video/features'][:]
            self.label = f['video/label'][:]
            # self.letter = f['video/letter'][:]
            #self.temporalletter = f['video/temporalletter'][:]
            # self.number = f['video/letternumber'][:]

    def __len__(self):
        return len(self.feature)

    def __getitem__(self, idx):
        feature = self.feature[idx]
        label = self.label[idx] 
        # letter = self.letter[idx].decode('utf-8') if isinstance(self.letter[idx], bytes) else self.letter[idx]
        # number = self.number[idx]
        #temporalletter = self.temporalletter[idx]
        return feature,label
    
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break

class BinaryClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=128, dropout_prob=0.2):
        super(BinaryClassifier, self).__init__()
        
        # First hidden layer
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.layer_norm1 = nn.LayerNorm(hidden_size)  
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_prob)
        
        # Second hidden layer (newly added)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.layer_norm2 = nn.LayerNorm(hidden_size // 2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_prob)

        # Third hidden layer (newly added)
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

        x = self.fc4(x)  # Output layer (no activation inside the model)
        return x
    
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    criterion = nn.BCEWithLogitsLoss()

    # Assuming you have 2 classes in the output
    num_classes = 2
    correct_per_class = [0] * num_classes
    total_per_class = [0] * num_classes

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # Sum up batch loss
            
            # Get predicted class
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # Track correct predictions for each class
            for i in range(num_classes):
                correct_per_class[i] += ((pred == i) & (target == i)).sum().item()
                total_per_class[i] += (target == i).sum().item()

    # Calculate average loss
    test_loss /= len(test_loader.dataset)

    # Calculate and print average class accuracy
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, total, 100. * correct / total))

    print("\nClass-wise accuracy:")
    for i in range(num_classes):
        if total_per_class[i] > 0:
            class_accuracy = 100. * correct_per_class[i] / total_per_class[i]
            print(f"Class {i}: {class_accuracy:.2f}%")
        else:
            print(f"Class {i}: No instances")



def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Custom Dataset Example')
    # parser.add_argument('--batch-size', type=int, default=64, metavar='N',
    #                     help='input batch size for training (default: 64)')
    # parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
    #                     help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()

    # Use CUDA if available
    device = torch.device("cuda" if not args.no_cuda and torch.cuda.is_available() else "cpu")
    print("device",device)


    torch.manual_seed(args.seed)

    train_file = "training1f384.h5"
    test_file = "testing1f384.h5"

    # Load the custom dataset
    training_data = BOBSLDataset(train_file)
    testing_data = BOBSLDataset(test_file)
    print(len(training_data))

    train_loader = DataLoader(training_data, batch_size=len(training_data))
    test_loader = DataLoader(testing_data, batch_size=len(testing_data))

    model = BinaryClassifier(384, 2).cuda()
    #.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    # if args.save_model:
    #     torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
