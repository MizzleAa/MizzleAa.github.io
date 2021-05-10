---
layout: post
title: "pytorch tensorboard"
date: 2021-04-22 16:00:00 +0000
categories: [python, pytorch, tensorboard, ai, code]
---

#### Pytorch tensorboard

##### python : 3.7.5

##### torch : 1.8.1+cu111

##### tensorboard : 2.5.0

     - 실행 방법 : tensorboard --logdir=runs

---

```python


import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import torchvision
import torchvision.transforms as transforms


def fun():

    torch.multiprocessing.freeze_support()
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ##############################################################################
    # transforms
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]
    )

    trainset = torchvision.datasets.FashionMNIST(
        './data',
        download=True,
        train=True,
        transform=transform
    )

    testset = torchvision.datasets.FashionMNIST(
        './data',
        download=True,
        train=False,
        transform=transform
    )

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=4,
        shuffle=True,
        num_workers=2
    )

    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=4,
        shuffle=False,
        num_workers=2
    )

    classes = (
        'T-shirt/top',
        'Trouser',
        'Rullover',
        'Dress',
        'Coat',
        'Snadal',
        'Shirt',
        'Sneaker',
        'Bag',
        'Ankle Boot'
    )
    ##############################################################################
    def imshow(img, one_channel=False):
        if one_channel:
            img = img.mean(dim=0)
        npimg = img.numpy()

        if one_channel:
            plt.imshow(npimg, cmap='Greys')
        else:
            plt.imshow(np.transpose(npimg, (1, 2, 0)))

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)

            self.fc1 = nn.Linear(16*4*4, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 16*4*4)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    net = Net()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        net.parameters(),
        lr=0.001,
        momentum=0.9
    )

    # tensorboard 설정
    writer = SummaryWriter('runs/fashion_mnist_experiment_1')

    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    img_grid = torchvision.utils.make_grid(images)

    imshow(img_grid, one_channel=True)

    # tensorbaord의 image에 image 추가
    writer.add_image('four_fashion_mnist_images', img_grid)
    # net및 images의 값을 추가
    writer.add_graph(net, images)

    def select_n_random(data, labels, n=100):
        assert len(data) == len(labels)
        perm = torch.randperm(len(data))
        return data[perm][:n], labels[perm][:n]
    # 100개의 데이터 임의추출
    images, labels = select_n_random(trainset.data, trainset.targets)
    class_labels = [classes[lab] for lab in labels]
    features = images.view(-1, 28 * 28)
    writer.add_embedding(
        features,
        metadata=class_labels,
        label_img=images.unsqueeze(1)
    )
    writer.close()

    def images_to_probs(net, images):
        '''
        학습된 신경망과 이미지 목록으로 부터 예측 결과 및 확률을 생성
        '''
        output = net(images)
        _, preds_tensor = torch.max(output, 1)
        preds = np.squeeze(preds_tensor.numpy())
        return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]

    def plot_classes_preds(net, images, labels):
        preds, probs = images_to_probs(net, images)
        fig = plt.figure(figsize=(12, 48))
        for idx in np.arange(4):
            ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
            imshow(images[idx], one_channel=True)
            ax.set_title("{0} , {1:.1f}%\n( label : {2} )".format(
                classes[preds[idx]],
                probs[idx] * 100.0,
                classes[labels[idx]],
            ),
                color=("green" if preds[idx] == labels[idx].item() else "red")
            )
        return fig

    # 학습 수행
    running_loss = 0.0
    for epoch in range(1):
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 1000 == 999:
                writer.add_scalar(
                    'training loss',
                    running_loss / 1000,
                    epoch * len(trainloader) + i
                )

                writer.add_figure(
                    'predictions vs. actuals',
                    plot_classes_preds(net, inputs, labels),
                    global_step=epoch * len(trainloader) + i
                )

                running_loss = 0.0

    class_probs = []
    class_preds = []

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            output = net(images)
            class_probs_batch = [F.softmax(el, dim=0) for el in output]
            _, class_preds_batch = torch.max(output, 1)

            class_probs.append(class_probs_batch)
            class_preds.append(class_preds_batch)

    test_probs = torch.cat([torch.stack(batch) for batch in class_probs])
    test_preds = torch.cat(class_preds)

    # 텐서보드에서 레이블 분류에 따른 학습율 확인
    def add_pr_curve_tensorboard(class_index, test_probs, test_preds, global_step=0):
        tensorboard_preds = test_preds == class_index
        tensorboard_probs = test_probs[:, class_index]

        writer.add_pr_curve(
            classes[class_index],
            tensorboard_preds,
            tensorboard_probs,
            global_step=global_step
        )

        writer.close()

    for i in range(len(classes)):
        add_pr_curve_tensorboard(i, test_probs, test_preds)

    print('end')


if __name__ == '__main__':
    fun()

```
