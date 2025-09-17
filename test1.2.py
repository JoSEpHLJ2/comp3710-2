import torch
import numpy as np
import matplotlib.pyplot as plt
import time

# ----------------------------
# 参数设置
# ----------------------------
N = 2048  # 样本点数
T = 1.0   # 信号持续时间
f0 = 1    # 基频
harmonics = [1, 3, 5, 20, 50]  # 用于 Fourier 重建的谐波
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# 1. PyTorch 版本 square_wave
# ----------------------------
def square_wave_torch(t):
    return torch.sign(torch.sin(2 * np.pi * f0 * t))

# ----------------------------
# 2. PyTorch Fourier 系列重建
# ----------------------------
def square_wave_fourier_torch(t, N_harmonics):
    t = t.to(device)
    result = torch.zeros_like(t, dtype=torch.float32, device=device)
    for k in range(N_harmonics):
        n = 2 * k + 1  # 奇次谐波
        result += torch.sin(2 * np.pi * n * f0 * t) / n
    return (4 / np.pi) * result

# ----------------------------
# 3. PyTorch GPU naive DFT
# ----------------------------
def naive_dft_torch(x):
    x = x.to(device).to(torch.complex64)  # 转复数类型
    N = x.shape[0]
    n = torch.arange(N, device=device).reshape(1, N)
    k = torch.arange(N, device=device).reshape(N, 1)
    W = torch.exp(-2j * np.pi * k * n / N)
    return torch.matmul(W, x)

# ----------------------------
# 创建时间向量
# ----------------------------
t_np = np.linspace(0.0, T, N, endpoint=False)
t = torch.tensor(t_np, dtype=torch.float32, device=device)

# ----------------------------
# 生成信号
# ----------------------------
square_torch = square_wave_torch(t)
square_fourier_torch = square_wave_fourier_torch(t, harmonics[-1])  # 最大谐波数

# ----------------------------
# 计时比较
# ----------------------------
# PyTorch naive DFT
start = time.time()
dft_torch = naive_dft_torch(square_fourier_torch)
end = time.time()
time_dft_torch = end - start

# PyTorch 内置 FFT
start = time.time()
fft_torch = torch.fft.fft(square_fourier_torch.to(torch.complex64))
end = time.time()
time_fft_torch = end - start

print(f"PyTorch naive DFT (GPU) : {time_dft_torch:.6f}s")
print(f"PyTorch FFT (GPU)       : {time_fft_torch:.6f}s")

# ----------------------------
# 可视化原始方波和 Fourier 重建
# ----------------------------
plt.figure(figsize=(12, 8))

# 原始方波
plt.subplot(2, 3, 1)
plt.plot(t_np, square_torch.cpu().numpy(), 'k', label="Square wave")
plt.title("Original Square Wave")
plt.ylim(-1.5, 1.5)
plt.grid(True)
plt.legend()

# Fourier 重建
for i, Nh in enumerate(harmonics, start=2):
    plt.subplot(2, 3, i)
    y = square_wave_fourier_torch(t, Nh)
    plt.plot(t_np, y.cpu().numpy(), label=f"N={Nh} harmonics")
    plt.plot(t_np, square_torch.cpu().numpy(), 'k--', alpha=0.5, label="Square wave")
    plt.title(f"Fourier Approximation with N={Nh}")
    plt.ylim(-1.5, 1.5)
    plt.grid(True)
    plt.legend()

plt.tight_layout()
plt.show()
