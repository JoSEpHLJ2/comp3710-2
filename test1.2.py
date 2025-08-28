import numpy as np
import matplotlib.pyplot as plt
import time

# ----------------------------
# 参数设置
# ----------------------------
N = 2048  # 样本点数
T = 1.0   # 信号持续时间
f0 = 1    # 基频
harmonics_count = 50  # 用多少谐波重建方波

# ----------------------------
# 方波函数
# ----------------------------
def square_wave(t):
    return np.sign(np.sin(2.0 * np.pi * f0 * t))

def square_wave_fourier(t, f0, N):
    result = np.zeros_like(t)
    for k in range(N):
        n = 2 * k + 1  # 奇次谐波
        result += np.sin(2 * np.pi * n * f0 * t) / n
    return (4 / np.pi) * result

# ----------------------------
# DFT 朴素实现
# ----------------------------
def naive_dft(x):
    N = len(x)
    X = np.zeros(N, dtype=np.complex128)
    for k in range(N):
        for n in range(N):
            angle = -2j * np.pi * k * n / N
            X[k] += x[n] * np.exp(angle)
    return X

# ----------------------------
# 生成信号
# ----------------------------
t = np.linspace(0.0, T, N, endpoint=False)
signal = square_wave_fourier(t, f0, harmonics_count)

# ----------------------------
# 计时比较 DFT vs FFT
# ----------------------------
start_time_naive = time.time()
dft_result = naive_dft(signal)
end_time_naive = time.time()
naive_duration = end_time_naive - start_time_naive

start_time_fft = time.time()
fft_result = np.fft.fft(signal)
end_time_fft = time.time()
fft_duration = end_time_fft - start_time_fft

print("\n--- DFT/FFT Performance Comparison ---")
print(f"Naïve DFT Execution Time: {naive_duration:.6f} seconds")
print(f"NumPy FFT Execution Time: {fft_duration:.6f} seconds")
if fft_duration > 0:
    print(f"FFT is approximately {naive_duration / fft_duration:.2f} times faster.")
else:
    print("FFT was too fast to measure a significant duration difference.")

print(f"\nOur DFT implementation is close to NumPy's FFT: {np.allclose(dft_result, fft_result)}")

# ----------------------------
# 频谱绘制
# ----------------------------
xf = np.fft.fftfreq(N, d=T/N)[:N//2]
magnitude = 2.0/N * np.abs(dft_result[0:N//2])

plt.style.use('seaborn-v0_8-darkgrid')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# 时域信号
ax1.plot(t, signal, color='c')
ax1.set_title(f'重建方波 (谐波数={harmonics_count})', fontsize=16)
ax1.set_xlabel('时间 (s)', fontsize=12)
ax1.set_ylabel('幅度', fontsize=12)
ax1.set_xlim(0, 1.0)
ax1.grid(True)

# 频域信号
ax2.stem(xf, magnitude, basefmt=" ")
ax2.set_title('DFT 幅度谱 (朴素 DFT)', fontsize=16)
ax2.set_xlabel('频率 (Hz)', fontsize=12)
ax2.set_ylabel('幅度', fontsize=12)
ax2.set_xlim(0, 50)
ax2.grid(True)

# 标注前几个奇次谐波
for i in range(1, 20, 2):  # 只标奇数
    if i < len(xf):
        ax2.axvline(xf[i], color='r', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()
