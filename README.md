# PBGCA -  Physically based galaxy clustering algorithm

Agglomenative clustering algorithm for galaxy clusters considering physical form of galaxy's structures

Clustering algoritm's initial purpose of which was to solve galaxy clustering task. But it can be scaled to solve simillar problem of finding clusters elongated to center of coordinates grid in N-dimentional grid

### Installation
```
pip install pbgca
```

### Get started
How to multiply one number by another with this lib:

```Python
from pbgca import Clusterer as PBGCA

#Get your data
data = pd.read_csv('some_data.csv')

# Instantiate a PBGCA object
clusterer=PBGCA()

# Call fit method
clusterer.fit(data)

# Get result of clustering
result = clusterer.labels_
```


Input Example:
  3d coordinates of galaxies in form of np matrix or pandas dataframe (here units: Mpc)
  Example:

| x      | y | z    |
| ------------- | ------------- |------------- |
| 1      | 34       | 0.45   |
| -1   | 94        | -0.322      |
| ..   | ...       | ...      |


