import pandas as pd
import numpy as np
from mimaslib.clusterer import *
from mimaslib.fancyplot import *

coor=pd.read_csv('ra_dec_z_data.csv')
tmp_coor=coor[['x','y','z']]
tmp_coor=tmp_coor[tmp_coor.x.between(10, 30)]
tmp_coor=tmp_coor[tmp_coor.y.between(-10, 10)]
tmp_coor=tmp_coor[tmp_coor.z.between(-10, 10)]

def test_some():
    print(tmp_coor.shape)
    clusterer=Clusterer(init_cluster_length=5,coefs = 1+1/arange(1,100)[10:],max_iter = 30)
    labels=clusterer.fit(tmp_coor[['x','y','z']])
    get_clusters_plot(clusterer.clusters)
    np.save('labes.npy', labels)
    assert 1 == 1