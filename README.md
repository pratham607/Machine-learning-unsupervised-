So first we selected the features for clustering:

X = df[["Age","Spending"]]


Here we are telling the model that only Age and Spending will be used to form clusters.

Then:

model = KMeans(n_clusters=2, random_state=42, n_init=10)


We are creating a KMeans model and telling it:

Make 2 clusters

Use random_state so result stays same

Try 10 times internally and choose the best clustering

After that:

df["Group"] = model.fit_predict(X)


Here the actual decision happens.

Based on Age and Spending values, the model calculates distances, forms clusters, adjusts centroids again and again until stable, and then assigns each row either 0 or 1.

So now the clustering is already done.

The Group column just reflects what the model decided.

It’s not random. It’s based on distance from centroids.

Then comes this part:

for group in df["Group"].unique():


Here we are basically saying:

Take each unique cluster number (in this case 0 and 1).

Since we used 2 clusters, unique values are 0 and 1.

Then:

group_data = df[df["Group"] == group]


This line just separates the rows based on cluster number.

If group = 0 → it takes only rows where Group is 0
If group = 1 → it takes only rows where Group is 1

Then:

plt.scatter(...)


It plots each cluster separately.

So overall:

Clustering already happened in fit_predict

The loop is just reading those cluster labels

It separates them and plots them

The 0 and 1 are just cluster labels decided by KMeans based on Age and Spending.
