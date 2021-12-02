# K means clustering (customer segmentation project)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# importing the dataset

data = pd.read_csv('E:\\datasets\\Mall_Customers.csv')
#print(data)

x = data.iloc[:, 3:5].values
#print(x)

# using the elbow method for choosing the optimal number of clusters
from sklearn.cluster import KMeans
WCSS = []
for i in range(1, 11):
    model = KMeans(n_clusters= i, init='k-means++')
    model.fit(x)
    WCSS.append(model.inertia_)


n_clusters = [*range(1, 11)]

plt.plot(n_clusters, WCSS)
plt.show()

# from the elbow method, we know that the right number of clusters (5)

clusters = KMeans(n_clusters= 5, init='k-means++')
clusters.fit(x)
y = clusters.fit_predict(x)


# visualization
plt.scatter(x[y == 0, 0], x[y == 0, 1], c='r')
plt.scatter(x[y == 1, 0], x[y == 1, 1], c='k')
plt.scatter(x[y == 2, 0], x[y == 2, 1], c='green')
plt.scatter(x[y == 3, 0], x[y == 3, 1], c='y')
plt.scatter(x[y == 4, 0], x[y == 4, 1], c='m')
plt.show()

# new data point
income = int(input("enter your income: "))
spending = int(input("enter your spending score: "))
new_point = [[income, spending]]
your_group = clusters.predict(new_point)
print(your_group)

