import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
data = {
    "Customer " :["Riya","Aman","Faizan","Neha","Imran","Senha"],
    "Age":[20,30,40,22,38,25],
    "Spending":[100,200,300,110,290,130]

}
df=pd.DataFrame(data)
X=df[["Age","Spending"]]
model=KMeans(n_clusters=2,random_state=92,n_init=10)
df["Group"]=model.fit_predict(X)

plt.figure(figsize=(6,5))
for group in df["Group"].unique():
    group_data = df[df["Group"]==group]
    plt.scatter(group_data["Age"],group_data["Spending"], label=f'Group {group}')
plt.xlabel("Age")
plt.ylabel("Spending")
plt.title("Customer Segmentation")
plt.legend()
plt.show()