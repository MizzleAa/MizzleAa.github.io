---
layout: post
title: "pytorch tensor"
date: 2021-04-22 16:00:00 +0000
categories: [python, pytorch, ai, code]
---

### Pytorch tensor

#### python : 3.7.5

#### torch : 1.8.1+cu111

---

```python

import torch
import numpy as np

def fun():
    torch.multiprocessing.freeze_support()
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Directly from data
    data = [
        [1, 2],
        [3, 4]
    ]
    x_data = torch.tensor(data)
    print(f"x_data = {x_data}")

    # From a NumPy array
    np_array = np.array(data)
    x_np = torch.from_numpy(np_array)

    print(f"x_np = {x_np}")

    # From another tensor:
    x_ones = torch.ones_like(x_data)
    x_rand = torch.rand_like(x_data, dtype=torch.float)

    print(f"x_ones = {x_ones}")
    print(f"x_rand = {x_rand}")

    # With random or constant values:
    shape = (2, 3,)
    rand_tensor = torch.rand(shape)
    ones_tensor = torch.ones(shape)
    zeros_tensor = torch.zeros(shape)

    print(f"rand_tensor = {rand_tensor}")
    print(f"ones_tensor = {ones_tensor}")
    print(f"zeros_tensor = {zeros_tensor}")

    # Attributes of a Tensor
    tensor = torch.rand(3, 4)
    print(f"tensor.shape = {tensor.shape}")
    print(f"tensor.dtype = {tensor.dtype}")
    print(f"tensor.device = {tensor.device}")

    # Operations on Tensors
    if torch.cuda.is_available():
        tensor = tensor.to('cuda')

    tensor = torch.ones(4, 4)
    print(f"tensor[0] = {tensor[0]}")
    print(f"tensor[:,0] = {tensor[:,0]}")
    print(f"tensor[...,-1] = {tensor[...,-1]}")
    print(f"tensor = {tensor}")
    tensor[1, :] = 0
    print(f"tensor = {tensor}")

    # Joining tensors
    tensor_cat = torch.cat([tensor, tensor, tensor], dim=0)
    print(f"tensor_cat(↧) = {tensor_cat}")
    tensor_cat = torch.cat([tensor, tensor, tensor], dim=1)
    print(f"tensor_cat(↦) = {tensor_cat}")

    # Arithmetic operations
    y1 = tensor @ tensor.T
    print(f"tensor @ tensor.T = {y1}")

    y2 = tensor.matmul(tensor.T)
    print(f"tensor.matmul(tensor.T) = {y2}")

    y3 = torch.rand_like(tensor)
    torch.matmul(tensor, tensor.T, out=y3)
    print(f"torch.matmul(tensor, tensor.T, out=y3) = {y3}")

    z1 = tensor * tensor
    print(f"tensor * tensor = {z1}")

    z2 = tensor.mul(tensor)
    print(f"tensor.mul(tensor) = {z2}")

    z3 = torch.rand_like(tensor)
    torch.mul(tensor, tensor, out=z3)
    print(f"tensor.mul(tensor) = {z3}")

    # Single-element tensors
    agg = tensor.sum()
    print(f"tensor.sum() = {agg}")

    agg_item = agg.item()
    print(f"tensor.sum().item() = {agg_item}, {type(agg_item)}")

    # In-place operations
    tensor_out = tensor.add(5)
    print(f"tensor.add(5) = {tensor_out}")

    tensor.add_(5)
    print(f"tensor.add_(5) = {tensor}")

    # Tensor to NumPy array
    tensor = torch.ones(5)
    print(f"torch.ones(5) = {tensor}")

    numpy = tensor.numpy()
    print(f"tensor.numpy() = {numpy}")

    tensor.add_(1)
    print(f"tensor.add_(1) tensor = {tensor}")
    print(f"tensor.add_(1) numpy = {numpy}")

    # Numpy array to Tensor
    numpy = np.ones(5)
    print(f"np.ones(5) = {numpy}")

    tensor = torch.from_numpy(numpy)
    print(f"torch.from_numpy(numpy) = {tensor}")

    np.add(numpy, 3, out=numpy)
    print(f"np.add(numpy, 3, out=numpy) numpy = {numpy}")
    print(f"np.add(numpy, 3, out=numpy) tensor = {tensor}")


if __name__ == '__main__':
    fun()


```
