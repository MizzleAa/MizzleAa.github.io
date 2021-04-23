---
layout: post
title: "pytorch tutorial"
date: 2021-04-21 16:00:00 +0000
categories: [python, pytorch, ai, code]
---

### Pytorch tutorial

#### python : 3.7.5

#### torch : 1.8.1+cu111

---

```python

import torchvision
import torchvision.transforms as transforms

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt

import numpy as np


def fun():
    '''
    신경망의 일반적인 학습 과정은 다음과 같습니다:
    1. 학습 가능한 매개변수(또는 가중치(weight))를 갖는 신경망을 정의합니다.
    2. 데이터셋(dataset) 입력을 반복합니다.
    3. 입력을 신경망에서 전파(process)합니다.
    4. 손실(loss; 출력이 정답으로부터 얼마나 떨어져있는지)을 계산합니다.
    5. 변화도(gradient)를 신경망의 매개변수들에 역으로 전파합니다.
    6. 신경망의 가중치를 갱신합니다.
    - 일반적으로 다음과 같은 간단한 규칙을 사용합니다: 새로운 가중치(weight) = 가중치(weight) - 학습률(learning rate) * 변화도(gradient)

    '''
    ##############################################################################
    '''
    # torch를 사용하기 위해 multiprocessing 사용하는 프로그램이 고정되어(frozen) 윈도우 실행파일을 생성할때를 위한 지원을 추가
    # 윈도우가 아닌 다른 운영체제에서 실행 될때는 아무런 영향을 주지 않음
    # 모듈이 윈도우상의 파이썬 인터프리에터 의해 정상 실행시 효과를 받지 않음
    # 윈도우에서는 프로세싱에 대한 fork라는 기능 구현이 부족하기 때문에
    # 오직 윈도우에서 멀티프로세싱 구조를 쓰기 위해 사용
    '''
    torch.multiprocessing.freeze_support()
    ##############################################################################
    '''
    # torch를 gpu로 돌리기 위해 gpu 여부를 확인
    # gpu 존재시 cuda로 수행, 없을시 cpu로 수행
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ##############################################################################
    '''
    # Transforms
    # 대부분 신경망은 고정된 크기의 이미지라고 가정함으로, 데이터를 신경망에 입력하기전에 처리해야할 작업이 존재
    # Rescale : 이미지 크기 조절
    # RandomCrop : 이미지를 무작위로 자름 (data augmentation)
    # ToTensor : numpy이미지에서 torch이미지로 변경( 축변환 )
    # Normalize : 이미지가 가지는 픽셀값에 대해 Tensor로 변환시 0~1 사이의 값으로 바뀜 이에 따라 normalize를  수행하여 -1 ~ 1의 값으로 변경
    '''
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(
            mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]
    )

    ##############################################################################
    '''
    # torchvision에서 CIFAR10 데이터 셋을 불러온다.
    # train=True : 학습데이터로 불러옴
    # download=True : 데이터가 없을경우 해당 데이터를 받음
    # transform : 기존에 정의된 transforms값을 가져와 수행
    '''
    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    '''
    # datasets를 dataloader를 통해 결합을 수행
    # batch_size=4 : 로드할 배치당 샘플 수를 설정( 기본 = 1 )
    # shuffle=True : 모든 epoch에서 데이터를 다시 섞도록 설정( 기본 = False )
    # num_worker=2 : 데이터로드에 사용할 하위 프로세스 개수를 선언( 기본 = 0 )
    '''
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=4, shuffle=True, num_workers=2
    )

    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4, shuffle=False, num_workers=2
    )
    ##############################################################################
    '''
    # classes 품목에 대한 명칭을 지정(label)
    '''
    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )
    ##############################################################################
    '''
    # 신경망
    # nn.Module : 모든 신경망 모듈에 대한 기본 클레스
    # 모델도 해당 클레스의 하위클레스야야함
    # 모듈에는 다른 모듈이 포함될 수 있고, 또한 중첩이 가능함
    # 트리구조형태의 하위모듈을 할당이 가능
    '''
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            '''
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=1, #교차 배치의 간격을 조절
                padding=0, # 상하좌우에 대한 패딩을 조절
                dilation=1, # 커널에 따른 점 사이의 간격을 제어
                groups=1, # 입력-출력 사이의 연결 값 제어
                bias =True, # bias
                padding_mode='zeros' # 패딩 방법에 대한 정의 : 'zeros', 'reflect', 'replicate', 'circular'
            )
            '''
            self.conv1 = nn.Conv2d(3, 6, 5)
            '''
            nn.MaxPool2d(
                kernel_size, #커널 크기
                stride=None, #교차 배치의 간격을 조절
                padding=0 ,# 상하좌우에 대한 패딩을 조절
                dilation=1, # 커널에 따른 점 사이의 간격을 제어
                return_indices=False, # True일 경우 해당 출력과 함께 최대값을 반환 ( if True, will return the max indices along with the outputs. Useful for torch.nn.MaxUnpool2d later )
                ceil_mode=False # when True, will use ceil instead of floor to compute the output shape
            )
            '''
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)

            '''
            nn.Linear(
                in_features, # 입력 크기
                out_features, # 출력 크기
                bias = True # bias
            )
            '''
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            '''
            F = torch.nn.functional
            F.relu(
                input, # 입력 값
                inplace=False # can optionally do the operation in-place. Default: False
            )
            '''
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            '''
            view(
                *shape
                -1, #-1일경우 다른 차원에서 유추
            )
            # 자체 텐서와 동일한 데이터를 기준으로 새 탠서형식으로 모향을 변형시켜 반환
            '''
            x = x.view(-1, 16 * 5 * 5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    ##############################################################################
    '''
    # net 클레스 정의
    '''
    net = Net()
    '''
    # 사용할 장비(cpu/gpu)확인
    '''
    net.to(device)
    '''
    # logSoftmax 와 NLLLoss를 단일 클레스로 결합
    # n개의 클래스로 분류문제를 학습할때 사용
    # 불균형한 데이터 셋이 있을때 유용
    # 각 인수에 대한 가중치는 최소 1D Tensor이여야 함
    nn.CrossEntropyLoss(
        weight=None,
        size_average=None,
        ignore_index=-100,
        reduce=None,
        reduction='mean'
    )
    '''
    criterion = nn.CrossEntropyLoss()

    '''
    # momentum :
    # - 가중치를 갱신할 때 델타 규칙에 모멘텀을 추가로 더함
    # - 모멘텀을 사용하면 가중치 값이 바로 바뀌지 않고 어느 정도 일정한 방향을 유지하면서 이동
    # - 가속도처럼 같은 방향으로 더 많이 변화시켜 학습속도를 높여 빠른 학습을 수행

    optim.SGD(
        params, #매개변수 그룹 정의
        lr=<required parameter>, # 학습율
        momentum=0, # momentum factor
        dampening=0,  # dampening for momentum
        weight_decay=0, # weight decay
        nesterov=False # enables Nesterov momentum
    )
    '''
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    '''
    # 학습 수행
    # - 데이터셋을 수차례 반복
    '''
    for epoch in range(2):
        running_loss = 0.0

        for i, data in enumerate(trainloader, 0):
            '''
            # [inputs, labels]의 목록인 data로부터 입력을 받음
            # 마찬가지로 device를 설정
            '''
            # inputs, labels = data # cpu
            inputs, labels = data[0].to(device), data[1].to(device)  # gpu
            '''
            # 변화도(Gradient) 매개변수를 0으로 만듬
            # 일반적으로 memory footprint 가 적어 성능적 향상이 가능
            zero_grad(
                set_to_none=False
            )
            '''
            optimizer.zero_grad()
            '''
            # net을 통한 결과값 도출
            '''
            outputs = net(inputs)
            '''
            # backward : 모든 하위 클래스에 의해 재정의
            '''
            loss = criterion(outputs, labels)
            loss.backward()
            '''
            # 최적화 단계를 수행(가중치값 갱신)
            step(
                closure=None # 모델을 재평가하고 손실을 반환하는 closure
            )
            '''
            optimizer.step()

            '''
            # loss에 대한 통계치 계산
            '''
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    print("Finished Training")

    ##############################################################################
    # save model
    '''
    # model 저장
    '''
    PATH = "./cifar_net.pth"
    torch.save(net.state_dict(), PATH)

    ##############################################################################
    # load model
    '''
    # model 불러오기
    '''
    net = Net()
    net.load_state_dict(torch.load(PATH))
    net.to(device)
    ##############################################################################
    '''
    # 이미지를 보여주기 위한
    '''
    dataiter = iter(testloader)
    images, labels = dataiter.next()

    # 이미지를 출력
    def imshow(img):
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    imshow(torchvision.utils.make_grid(images))
    print("GroundTruth: ", " ".join("%5s" %
          classes[labels[j]] for j in range(4)))

    ##############################################################################
    '''
    # 학습율 확인
    '''
    class_correct = list(0.0 for i in range(10))
    class_total = list(0.0 for i in range(10))
    with torch.no_grad():
        for data in testloader:
            # images, labels = data
            images, labels = data[0].to(device), data[1].to(device)

            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print(
            "Accuracy of %5s : %2d %%"
            % (classes[i], 100 * class_correct[i] / class_total[i])
        )


if __name__ == "__main__":
    fun()



```
