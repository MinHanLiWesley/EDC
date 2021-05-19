from sklearn.decomposition import PCA, LatentDirichletAllocation

import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

tr_path = '/home/wesley/EDC/FPC/ML/training_data_FPC_V1_3m_5p_6t_21Clppm.csv'  # path to training data
# tt_path = 'covid.test.csv'   # path to testing data
training = pd.read_csv(tr_path)
pca_feats = list(range(2, 32))  # list(range(40, 57)) + list(range(58, 75)) + list(range(76, 93))
X_train = training.iloc[0:100,2:-4].to_numpy()
print(X_train)
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)

pca = PCA()
X_train_pca = pca.fit_transform(X_train_std)
import matplotlib.pyplot as plt
plt.bar(range(1, len(pca_feats)+1), pca.explained_variance_ratio_, alpha=0.5, align='center')
plt.step(range(1, len(pca_feats)+1), np.cumsum(pca.explained_variance_ratio_), where='mid')
plt.savefig("/home/wesley/EDC/FPC/ML/pca.png")