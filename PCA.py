import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
data = {
    "Age":[25,30,35,40,45,50],
    "Income":[30000,40000,50000,60000,70000,80000],
    "Savings":[1000,5000,8000,10000,15000,20000]
}
df=pd.DataFrame(data)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)
pca_df=pd.DataFrame(pca_result,columns=["pca1","pca2"])
explained_variance =pca.explained_variance_ratio_
print("explained variance")
print(np.round(explained_variance*100,2))
plt.figure(figsize=(8,6))
plt.scatter(pca_df["pca1"],pca_df["pca2"],color="black",s=80)
plt.title("PCA projection 2D")
plt.xlabel("PCA1 Main pattern")
plt.ylabel("PCA2 Minor pattern")
plt.show()