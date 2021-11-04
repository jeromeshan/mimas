import pandas as pd
import numpy as np
from mimaslib.clusterer import *
from mimaslib.fancyplot import *

coor=pd.read_csv('ra_dec_z_data.csv')
tmp_coor=coor[['x','y','z']].iloc[:1000000,:]
tmp_coor=tmp_coor[tmp_coor.x.between(-20, 20)]
tmp_coor=tmp_coor[tmp_coor.y.between(-20, 20)]
tmp_coor=tmp_coor[tmp_coor.z.between(-20, 20)]

def test_some():
    print(tmp_coor.shape)
    clusterer=Clusterer(init_cluster_length=1,coefs = 1+1/arange(1,100)[20:])
    labels=clusterer.fit(tmp_coor[['x','y','z']])
    get_clusters_plot(clusterer.clusters)
    np.save('labes.npy', labels)
    assert 1 == 2