import torch
import torch.nn as nn
from e2cnn.nn import R2Conv, GeometricTensor, FieldType, ReLU, InnerBatchNorm, SequentialModule, PointwiseMaxPool, \
    GroupPooling
from e2cnn.gspaces import Rot2dOnR2

import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader


class SO2Classifier(nn.Module):
    def __init__(self, num_classes=10, n_rotations=8):
        super(SO2Classifier, self).__init__()

        self.r2_act = Rot2dOnR2(N=n_rotations)

        self.input_type = FieldType(self.r2_act, 3 * [self.r2_act.trivial_repr])
        self.feature_type_64 = FieldType(self.r2_act, 64 * [self.r2_act.regular_repr])
        self.feature_type_128 = FieldType(self.r2_act, 128 * [self.r2_act.regular_repr])
        self.feature_type_256 = FieldType(self.r2_act, 256 * [self.r2_act.regular_repr])
        self.feature_type_512 = FieldType(self.r2_act, 512 * [self.r2_act.regular_repr])

        self.block1 = SequentialModule(
            R2Conv(self.input_type, self.feature_type_64, kernel_size=3, padding=1),
            InnerBatchNorm(self.feature_type_64),
            ReLU(self.feature_type_64),
            R2Conv(self.feature_type_64, self.feature_type_64, kernel_size=3, padding=1),
            InnerBatchNorm(self.feature_type_64),
            ReLU(self.feature_type_64),
            PointwiseMaxPool(self.feature_type_64, 2)
        )

        self.block2 = SequentialModule(
            R2Conv(self.feature_type_64, self.feature_type_128, kernel_size=3, padding=1),
            InnerBatchNorm(self.feature_type_128),
            ReLU(self.feature_type_128),
            R2Conv(self.feature_type_128, self.feature_type_128, kernel_size=3, padding=1),
            InnerBatchNorm(self.feature_type_128),
            ReLU(self.feature_type_128),
            PointwiseMaxPool(self.feature_type_128, 2)
        )

        self.block3 = SequentialModule(
            R2Conv(self.feature_type_128, self.feature_type_256, kernel_size=3, padding=1),
            InnerBatchNorm(self.feature_type_256),
            ReLU(self.feature_type_256),
            R2Conv(self.feature_type_256, self.feature_type_256, kernel_size=3, padding=1),
            InnerBatchNorm(self.feature_type_256),
            ReLU(self.feature_type_256),
            PointwiseMaxPool(self.feature_type_256, 2)
        )

        self.block4 = SequentialModule(
            R2Conv(self.feature_type_256, self.feature_type_512, kernel_size=3, padding=1),
            InnerBatchNorm(self.feature_type_512),
            ReLU(self.feature_type_512),
            R2Conv(self.feature_type_512, self.feature_type_512, kernel_size=3, padding=1),
            InnerBatchNorm(self.feature_type_512),
            ReLU(self.feature_type_512),
            PointwiseMaxPool(self.feature_type_512, 2)
        )

        self.block5 = SequentialModule(
            R2Conv(self.feature_type_512, self.feature_type_512, kernel_size=3, padding=1),
            InnerBatchNorm(self.feature_type_512),
            ReLU(self.feature_type_512),
            R2Conv(self.feature_type_512, self.feature_type_512, kernel_size=3, padding=1),
            InnerBatchNorm(self.feature_type_512),
            ReLU(self.feature_type_512),
            PointwiseMaxPool(self.feature_type_512, 2)
        )

        self.gpool = GroupPooling(self.feature_type_512)

        self.fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = GeometricTensor(x, self.input_type)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.gpool(x)
        x = x.tensor.view(x.tensor.size(0), -1)
        # print(x.shape)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # 数据增强
    transform = transforms.Compose([
        transforms.RandomRotation(degrees=(0, 360)),  # 随机旋转
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.ToTensor(),  # 转为Tensor
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),  # 归一化
    ])

    # 加载CIFAR-10数据集
    train_dataset = datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
    test_dataset = datasets.CIFAR10(root="./data", train=False, transform=transform, download=True)

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    classifier = SO2Classifier(num_classes=10)
    classifier.to(device)

    print("#params", sum(x.numel() for x in classifier.parameters()))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)

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
            torch.save(classifier.state_dict(), "so2_best_model.pth")
        print(f"Epoch [{epoch + 1}] Accuracy: {accuracy:.2f}% Best Accuracy: {best_accuracy:.2f}%")




