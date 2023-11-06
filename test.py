import torch

# 假设你有一个包含概率分布的 Tensor
probabilities = torch.tensor([0.2, 0.3, 0.5])

# 指定要采样的样本数量，例如 1
num_samples = 100

# 使用 torch.multinomial 进行采样
samples = torch.multinomial(probabilities, num_samples, replacement=True)

# 打印采样结果
print(samples)
