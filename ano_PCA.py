import numpy as np
from sklearn.decomposition import PCA






data = np.random.rand(10, 5)            # 生成10个样本，每个样本5个特征

pca = PCA(n_components=2)
low_dim_data = pca.fit_transform(data)  # 每个样本降为2维　
print(data)
print(low_dim_data)