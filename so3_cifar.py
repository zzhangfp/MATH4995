#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/11/24 19:55
# @Author  : CodeCat
# @File    : so3.py
# @Software: PyCharm
import gzip
import math
import pickle

import numpy as np
import torch
import torch.nn as nn
from e3nn import o3
from e3nn.nn import SO3Activation


def s2_near_identity_grid(max_beta: float = math.pi / 8, n_alpha: int = 8, n_beta: int = 3) -> torch.Tensor:
    """
    :return: rings around the north pole
    size of the kernel = n_alpha * n_beta
    """
    beta = torch.arange(1, n_beta + 1) * max_beta / n_beta
    alpha = torch.linspace(0, 2 * math.pi, n_alpha + 1)[:-1]
    a, b = torch.meshgrid(alpha, beta, indexing="ij")
    b = b.flatten()
    a = a.flatten()
    return torch.stack((a, b))


def so3_near_identity_grid(
    max_beta: float = math.pi / 8, max_gamma: float = 2 * math.pi, n_alpha: int = 8, n_beta: int = 3, n_gamma=None
) -> torch.Tensor:
    """
    :return: rings of rotations around the identity, all points (rotations) in
    a ring are at the same distance from the identity
    size of the kernel = n_alpha * n_beta * n_gamma
    """
    if n_gamma is None:
        n_gamma = n_alpha  # similar to regular representations
    beta = torch.arange(1, n_beta + 1) * max_beta / n_beta
    alpha = torch.linspace(0, 2 * math.pi, n_alpha)[:-1]
    pre_gamma = torch.linspace(-max_gamma, max_gamma, n_gamma)
    A, B, preC = torch.meshgrid(alpha, beta, pre_gamma, indexing="ij")
    C = preC - A
    A = A.flatten()
    B = B.flatten()
    C = C.flatten()
    return torch.stack((A, B, C))


def s2_irreps(lmax: int) -> o3.Irreps:
    return o3.Irreps([(1, (l, 1)) for l in range(lmax + 1)])


def so3_irreps(lmax: int) -> o3.Irreps:
    return o3.Irreps([(2 * l + 1, (l, 1)) for l in range(lmax + 1)])


def flat_wigner(lmax: int, alpha: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    return torch.cat([(2 * l + 1) ** 0.5 * o3.wigner_D(l, alpha, beta, gamma).flatten(-2) for l in range(lmax + 1)], dim=-1)


class S2Convolution(torch.nn.Module):
    def __init__(self, f_in, f_out, lmax, kernel_grid) -> None:
        super().__init__()
        self.register_parameter(
            "w", torch.nn.Parameter(torch.randn(f_in, f_out, kernel_grid.shape[1]))
        )  # [f_in, f_out, n_s2_pts]
        self.register_buffer(
            "Y", o3.spherical_harmonics_alpha_beta(range(lmax + 1), *kernel_grid, normalization="component")
        )  # [n_s2_pts, psi]
        self.lin = o3.Linear(s2_irreps(lmax), so3_irreps(lmax), f_in=f_in, f_out=f_out, internal_weights=False)

    def forward(self, x):
        psi = torch.einsum("ni,xyn->xyi", self.Y, self.w) / self.Y.shape[0] ** 0.5
        return self.lin(x, weight=psi)


class SO3Convolution(torch.nn.Module):
    def __init__(self, f_in, f_out, lmax, kernel_grid) -> None:
        super().__init__()
        self.register_parameter(
            "w", torch.nn.Parameter(torch.randn(f_in, f_out, kernel_grid.shape[1]))
        )  # [f_in, f_out, n_so3_pts]
        self.register_buffer("D", flat_wigner(lmax, *kernel_grid))  # [n_so3_pts, psi]
        self.lin = o3.Linear(so3_irreps(lmax), so3_irreps(lmax), f_in=f_in, f_out=f_out, internal_weights=False)

    def forward(self, x):
        psi = torch.einsum("ni,xyn->xyi", self.D, self.w) / self.D.shape[0] ** 0.5
        return self.lin(x, weight=psi)


class SO3Classifier(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        f1 = 64
        f2 = 128
        f3 = 256
        f_output = 10

        b_in = 60
        lmax1 = 20

        b_l1 = 20
        lmax2 = 10

        b_l2 = 10
        lmax3 = 5

        b_l4 = 6

        grid_s2 = s2_near_identity_grid()
        grid_so3 = so3_near_identity_grid()

        self.from_s2 = o3.FromS2Grid((b_in, b_in), lmax1)

        self.conv1 = S2Convolution(1, f1, lmax1, kernel_grid=grid_s2)

        self.act1 = SO3Activation(lmax1, lmax2, torch.relu, b_l1)

        self.conv2 = SO3Convolution(f1, f2, lmax2, kernel_grid=grid_so3)

        self.act2 = SO3Activation(lmax2, lmax3, torch.relu, b_l2)

        self.conv3 = SO3Convolution(f2, f3, lmax3, kernel_grid=grid_so3)

        self.act3 = SO3Activation(lmax3, 0, torch.relu, b_l4)

        self.fc = nn.Sequential(
            nn.Linear(f3, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, f_output),
        )

        # self.w_out = torch.nn.Parameter(torch.randn(f2, f_output))

    def forward(self, x):
        x = x.transpose(-1, -2)  # [batch, features, alpha, beta] -> [batch, features, beta, alpha]
        x = self.from_s2(x)  # [batch, features, beta, alpha] -> [batch, features, irreps]
        x = self.conv1(x)  # [batch, features, irreps] -> [batch, features, irreps]
        x = self.act1(x)  # [batch, features, irreps] -> [batch, features, irreps]
        x = self.conv2(x)  # [batch, features, irreps] -> [batch, features, irreps]
        x = self.act2(x)  # [batch, features, irreps] -> [batch, features, irreps]
        x = self.conv3(x) # [batch, features, irreps] -> [batch, features, irreps]
        x = self.act3(x) # [batch, features, irreps] -> [batch, features, irreps]
        x = x.squeeze()  # [batch, features,]
        x = self.fc(x)

        return x


MNIST_PATH = "s3_cifar.gz"
DEVICE = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 1e-3


def load_data(path, batch_size):
    with gzip.open(path, "rb") as f:
        dataset = pickle.load(f)
    # print(len(dataset["train"]["images"]))
    # print(len(dataset["train"]["labels"]))
    train_data = torch.from_numpy(np.array(dataset["train"]["images"][:, None, :, :])).to(torch.float32)
    train_labels = torch.from_numpy(np.array(dataset["train"]["labels"])).to(torch.long)

    # train_data /= 57  This normalization was hurtful, see @dmklee comment in discussions/344
    # print(len(train_data))
    # print(len(train_labels))
    train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_data = torch.from_numpy(np.array(dataset["test"]["images"][:, None, :, :])).to(torch.float32)
    test_labels = torch.from_numpy(np.array(dataset["test"]["labels"])).to(torch.long)

    # test_data /= 57

    test_dataset = torch.utils.data.TensorDataset(test_data, test_labels)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader, train_dataset, test_dataset


def main() -> None:
    train_loader, test_loader, train_dataset, _ = load_data(MNIST_PATH, BATCH_SIZE)

    classifier = SO3Classifier()
    classifier.to(DEVICE)

    print("#params", sum(x.numel() for x in classifier.parameters()))

    optimizer = torch.optim.Adam(classifier.parameters(), lr=LEARNING_RATE)
    best_accuracy = 0.0

    for epoch in range(NUM_EPOCHS):
        for i, (images, labels) in enumerate(train_loader):
            classifier.train()

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = classifier(images)
            loss = torch.nn.functional.cross_entropy(outputs, labels)
            loss.backward()

            optimizer.step()

            print(
                "\rEpoch [{0}/{1}], Iter [{2}/{3}] Loss: {4:.4f}".format(
                    epoch + 1, NUM_EPOCHS, i + 1, len(train_dataset) // BATCH_SIZE, loss.item()
                ),
                end="",
            )
        print("")
        correct = 0
        total = 0
        for images, labels in test_loader:
            classifier.eval()

            with torch.no_grad():
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = classifier(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(classifier.state_dict(), "so3_cifar_best_model.pth")
        print(f"Test Accuracy: {100 * correct / total}")


if __name__ == "__main__":
    main()
