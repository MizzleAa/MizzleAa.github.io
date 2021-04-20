---
layout: post
title: "python ai pytorch install"
date: 2021-04-18 21:00:00 +0000
categories: [python, ai, install]
---

### Python Pytorch Install with GPU

#### version : 3.7.5

- 해당 기준은 python을 기준으로 설치하는 버전입니다.
- conda를 사용하지 않았으며, 가상환경 생성을 통해 해당 python package를 설치합니다.

---

#### 1. gpu : <https://en.wikipedia.org/wiki/CUDA#Version_features_and_specifications>

- 자기 버전에 맞는 gpu 확인

#### 2. cuda : <https://developer.nvidia.com/cuda-toolkit-archive>

- gpu버전에 해당하는 cuda 설치
- nvidia-smi : CUDA Version 확인

#### 3. pytorch : <https://pytorch.org/get-started/locally/>

- build 확인, os 확인, package 확인, language 확인

#### 4. 각자 환경에 맞는 버전을 설치

- 작성자 환경 : stable(1.8.1) > Windows > Pip > Python > CUDA 11.1
  > pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

#### 5. 가상환경(venv) 생성

- 프로잭트 생성 위치 or 가상환경 관리 위치
  > python -m venv venv  
  > venv\Scripts\activate  
  > python -m pip install --upgrade pip

#### 6. 4.의 "pip3 ~" 문구 내용을 콘솔에 붙여 넣기

- 완료 때 까지 시간 소요

#### 7. pytorch 동작 확인 예제 코드 : <https://tutorials.pytorch.kr/beginner/pytorch_with_examples.html>

```python

import torch

dtype = torch.float
#device = torch.device("cpu") # cpu
device = torch.device("cuda:0") # gpu

# N은 배치 크기이며, D_in은 입력의 차원입니다;
# H는 은닉층의 차원이며, D_out은 출력 차원입니다.
N, D_in, H, D_out = 64, 1000, 100, 10

# 무작위의 입력과 출력 데이터를 생성합니다.
x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

# 무작위로 가중치를 초기화합니다.
w1 = torch.randn(D_in, H, device=device, dtype=dtype)
w2 = torch.randn(H, D_out, device=device, dtype=dtype)

learning_rate = 1e-6
for t in range(500):
    # 순전파 단계: 예측값 y를 계산합니다.
    h = x.mm(w1)
    h_relu = h.clamp(min=0)
    y_pred = h_relu.mm(w2)

    # 손실(loss)을 계산하고 출력합니다.
    loss = (y_pred - y).pow(2).sum().item()
    if t % 100 == 99:
        print(t, loss)

    # 손실에 따른 w1, w2의 변화도를 계산하고 역전파합니다.
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2.t())
    grad_h = grad_h_relu.clone()
    grad_h[h < 0] = 0
    grad_w1 = x.t().mm(grad_h)

    # 경사하강법(gradient descent)를 사용하여 가중치를 갱신합니다.
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
```

#### 8. 실행 결과

> 99 177.5563507080078  
> 199 0.22159364819526672  
> 299 0.0007461378700099885  
> 399 4.888954572379589e-05  
> 499 1.6039884940255433e-05
