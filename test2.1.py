import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# ----------------------------
# 1. 加载 LFW 数据
# ----------------------------
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
n_samples, h, w = lfw_people.images.shape
X = lfw_people.data
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

print("数据集规模:")
print("n_samples:", n_samples)
print("n_features:", X.shape[1])
print("n_classes:", n_classes)

# ----------------------------
# 2. 划分训练集/测试集
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# ----------------------------
# 3. 转为 PyTorch 张量
# ----------------------------
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

# ----------------------------
# 4. PCA / Eigenfaces
# ----------------------------
n_components = 150

# 数据中心化
mean = torch.mean(X_train_tensor, dim=0)
X_train_centered = X_train_tensor - mean
X_test_centered = X_test_tensor - mean

# SVD 分解
U, S, Vh = torch.linalg.svd(X_train_centered, full_matrices=False)
components = Vh[:n_components]
eigenfaces = components.reshape((n_components, h, w))

# 投影到 PCA 子空间
X_train_pca = X_train_centered @ components.T
X_test_pca = X_test_centered @ components.T

print("训练集 PCA 形状:", X_train_pca.shape)
print("测试集 PCA 形状:", X_test_pca.shape)

# ----------------------------
# 5. 可视化 Eigenfaces
# ----------------------------
def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].detach().numpy(), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())

eigenface_titles = [f"Eigenface {i}" for i in range(n_components)]
plot_gallery(eigenfaces, eigenface_titles, h, w)
plt.show()

# ----------------------------
# 6. 累计解释方差（紧凑性）
# ----------------------------
explained_variance = (S**2) / (n_samples - 1)
total_var = torch.sum(explained_variance)
explained_variance_ratio = explained_variance / total_var
ratio_cumsum = torch.cumsum(explained_variance_ratio, dim=0)

plt.plot(torch.arange(n_components), ratio_cumsum[:n_components].detach().numpy())
plt.title('PCA 累计解释方差（紧凑性）')
plt.xlabel('主成分数量')
plt.ylabel('累计解释方差')
plt.show()

# ----------------------------
# 7. 随机森林分类
# ----------------------------
# 转回 NumPy 数组用于 sklearn
X_train_pca_np = X_train_pca.numpy()
X_test_pca_np = X_test_pca.numpy()

estimator = RandomForestClassifier(n_estimators=150, max_depth=15, max_features=150)
estimator.fit(X_train_pca_np, y_train)
predictions = estimator.predict(X_test_pca_np)

correct = predictions == y_test
accuracy = np.sum(correct) / len(y_test)

print("测试集总数:", len(y_test))
print("预测正确数量:", np.sum(correct))
print("准确率:", accuracy)
print(classification_report(y_test, predictions, target_names=target_names))
