import os
import trimesh
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.autograd import Variable
import e3nn.nn


class ModelNet10Dataset(Dataset):
    def __init__(self, data_dir, num_points=2048, split='train'):
        self.data_dir = data_dir
        self.num_points = num_points
        self.split = split
        self.files = []
        self.labels = []
        self.load_data()

    def load_data(self):
        classes = sorted(os.listdir(self.data_dir))
        for label, cls in enumerate(classes):
            cls_dir = os.path.join(self.data_dir, cls)
            for file in os.listdir(os.path.join(cls_dir, self.split)):
                if file.endswith('.off'):
                    self.files.append(os.path.join(cls_dir, self.split, file))
                    self.labels.append(label)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        mesh = trimesh.load_mesh(self.files[idx])
        points = mesh.sample(self.num_points)
        label = self.labels[idx]
        return torch.tensor(points, dtype=torch.float32), label


class STN3d(nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.to(x.device)
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class PointNetEncoder(nn.Module):
    def __init__(self, global_feat=True, channel=3):
        super(PointNetEncoder, self).__init__()
        self.stn = STN3d(channel)
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 512, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(512)
        self.global_feat = global_feat

    def forward(self, x):
        B, D, N = x.size()
        trans = self.stn(x)
        x = x.transpose(2, 1)
        if D > 3:
            feature = x[:, :, 3:]
            x = x[:, :, :3]
        x = torch.bmm(x, trans)
        if D > 3:
            x = torch.cat([x, feature], dim=2)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 512)
        return x


# 定义网络架构
class SO3Network(nn.Module):
    def __init__(self, num_classes=10):
        super(SO3Network, self).__init__()
        self.input_features = e3nn.nn.FullyConnectedNet([3, 10, 3], torch.relu)
        self.feat = PointNetEncoder(global_feat=True, channel=3)
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        # x是形状为(batch_size, num_points, 3)的点云数据
        # 先将点云数据转换为球谐函数特征
        x = self.input_features(x)
        x = x.permute(0, 2, 1).contiguous()
        x = self.feat(x)
        x = self.fc(x)

        return x


if __name__ == '__main__':
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    train_dataset = ModelNet10Dataset(data_dir='ModelNet10', split='train')
    test_dataset = ModelNet10Dataset(data_dir='ModelNet10', split='test')

    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    classifier = SO3Network().to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 100

    best_accuracy = 0.0

    for epoch in range(num_epochs):
        classifier.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = classifier(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch + 1}], Step [{i + 1}], Loss: {running_loss / 100:.4f}")
                running_loss = 0.0

        correct = 0
        total = 0
        with torch.no_grad():
            classifier.eval()
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = classifier(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(classifier.state_dict(), "so3_ModelNet10_best_model.pth")
        print(f"Epoch [{epoch + 1}] Accuracy: {accuracy:.2f}% Best Accuracy: {best_accuracy:.2f}%")
