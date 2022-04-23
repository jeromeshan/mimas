from .cluster import Cluster
from typing import List
import ray
from scipy.spatial import Delaunay
from numpy import transpose, arange, concatenate, eye
import numpy as np
import time
from .cm_python import connectivity_matrix_python
from cm_cython import connectivity_matrix_cython
from joblib import Parallel, delayed, cpu_count


class ConnectivityMatrix:
    """
    Compute connectivity matrix between clusters
    """

    def __init__(self, clusters:List[Cluster],epsilon,limit_radian,n_dim,cython,parallel):
        """Create instance of ConnectivityMatrix class. Extract all necessary data from clusters and into array

        :param clusters: List of Clusters
        :type clusters: List[Cluster]
        :param epsilon: epsilon
        :type epsilon: float
        :param limit_radian: limit radian distance between clusters
        :type limit_radian: int
        :param n_dim: number of dimensions
        :type n_dim: int
        :param parallel: use cython implementation 
        :type parallel: bool
        :param parallel: use parallel computing 
        :type parallel: string (joblib or ray)
        """

        # start = time.time()
        self.data = []
        self.epsilon = epsilon
        self.limit_radian = limit_radian
        self.parallel = parallel
        self.cython = cython

        for cluster in clusters:
            self.data.append([cluster.centroid.reshape(1,-1),#0
            len(cluster.galaxies),#1
            cluster.get_length(),#2
            cluster.rotated_cube,#3
            cluster.galaxies[:,:n_dim],
            cluster.isWasComplete()
            ])
        self.data = np.array(self.data,dtype=object)

        np.save('cm2.npy',self.data)
        # print('\t\t','read data time: ', time.time()-start, ' s')

    
    def split_array(self, a, n):
        """Utility method. Splits array into n equal parts

        :param a: array to split
        :type a: array
        :param n: number of splits
        :type n: int
        :return: array ``a`` splited into ``n`` parts
        :rtype: array
        """
        k, m = divmod(len(a), n)
        return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

    def get_matrix(self):
        """Returs connectivity matrix

        :return: Connectivity matrix
        :rtype: array
        """
        if self.parallel == 'ray':
            ray.init()
            data = ray.put(self.data) 
            graph = ray.get([self.parallel_connectivity_matrix.remote(self, data, subset,self.epsilon,self.limit_radian,self.cython) for subset in self.split_array(arange(len(self.data)).astype(int), 64)])
            graph = concatenate( graph, axis = 0 )
            ray.shutdown()
            
        elif self.parallel == 'joblib':

            graph = Parallel(n_jobs = (-1))(delayed(self.connectivity_matrix)(self.data,subset,self.epsilon,self.limit_radian,self.cython) for subset in self.split_array(arange(len(self.data)).astype(int), 64))
            graph = concatenate( graph, axis = 0 )
        else:
            graph = self.connectivity_matrix(self.data,arange(len(self.data)).astype(int),self.epsilon,self.limit_radian,self.cython)
    
        graph = graph+transpose(graph)
        graph = graph-eye(len(self.data))
        return graph



    def connectivity_matrix(self, data , rows, epsilon,limit_radian, cython):
        """Utility method for connectivity matrix. Compute part of matrix

        :param data: data of clusters
        :type data: array
        :param rows: rows to compute
        :type rows: array
        :param epsilon: epsilon
        :type epsilon: float
        :param limit_radian: limit radian
        :type limit_radian: float
        :param cython: if using cython
        :type cython: bool
        :return: part of connectivity matrix
        :rtype: array
        """
        if(cython):
            cm =  connectivity_matrix_cython(data,rows,self.epsilon,self.limit_radian)
            return cm
        else:
            cm = connectivity_matrix_python(data,rows,self.epsilon,self.limit_radian)
            return cm


    @ray.remote
    def parallel_connectivity_matrix(self, data, rows,epsilon,limit_radian,cython):
        """Utility method for connectivity matrix. Compute part of matrix in parallel using ray


        :param data: data of clusters
        :type data: array
        :param rows: rows to compute
        :type rows: array
        :param epsilon: epsilon
        :type epsilon: float
        :param limit_radian: limit radian
        :type limit_radian: float
        :param cython: if using cython
        :type cython: bool
        :return: connectivity matrix
        :rtype: array
        """
        return self.connectivity_matrix(data, rows,epsilon,limit_radian,cython)


