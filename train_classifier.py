import time
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim import Adam

from dataset import VideoDataset, get_dataset_paths
from video_classifier import VideoClassifier

def main():

    train_mode = True
    preload = False

    if preload:
        model_filepath = 'models/video_classifierResRNN_v1.pth'

    full_dataset, class_to_idx, idx_to_class = get_dataset_paths('data', labels=True)

    resNet18_transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_data, test_data = split_dataset(full_dataset, random_state=42)
    train_dataset = VideoDataset(train_data, transform=resNet18_transform)
    test_dataset = VideoDataset(test_data, transform=resNet18_transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=8, pin_memory=True)

    model = VideoClassifier()

    if preload:
        model.load_state_dict(torch.load(model_filepath))

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if train_mode:
        train_model(model, 50, train_loader, device, criterion, optimizer)
        torch.save(model.state_dict(), 'models/video_classifierResRNN_v1.pth')

    test_model(model, test_loader, device)


def split_dataset(dataset, test_size=0.1, random_state=None):
    data = [data for data, _ in dataset]
    labels = [label for _, label in dataset]

    train_paths, test_paths, train_labels, test_labels = train_test_split(
        data, labels, test_size=test_size, stratify=labels, random_state=random_state
    )

    return list(zip(train_paths, train_labels)), list(zip(test_paths, test_labels))


def train_model(net, n_epochs, loader, device, criterion, optimizer):
    scaler = GradScaler()
    net.to(device)

    for epoch in range(n_epochs):
        start_time = time.time()

        for i, data in enumerate(loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            with autocast():
                outputs = net(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if (i + 1) % 100 == 0:
                print(f'Epoch {epoch}, Iteration {i + 1}, Loss: {loss.item()}, Time per 100 iterations: {time.time() - start_time}')
                start_time = time.time()

        torch.save(net.state_dict(), f'models/video_classifierResRNN_v1_{epoch + 1}.pth')

    print('Finished Training')

def test_model(model, loader, device):
    model.to(device)
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data in loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            with autocast():
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the network on the test set videos: {accuracy:.2f}%')

    return accuracy

if __name__ == '__main__':
    main()