import torch
import numpy as np

# 类型转换
np_data = np.arange(6).reshape((2, 3))
np2torch = torch.from_numpy(np_data)
torch2np = np2torch.numpy()

print(
    f'\nnumpy\n{np_data}',
    f'\ntorch\n{np2torch}',
    f'\ntorch\n{torch2np}'
)


# 用法类似
data = [[-1, -2], [1, 2]]
tensor = torch.FloatTensor(data)

print(
    '\nabs',
    f'\nnp\n{np.abs(data)}',
    f'\ntorch\n{torch.abs(tensor)}'
)