from .cluster import Cluster
from typing import List
import ray
from scipy.spatial import Delaunay
from numpy import transpose, array, arange, concatenate, eye, unique, dot, zeros, append, ones
import numpy as np
from numpy.linalg import norm
from scipy.spatial import distance
from scipy.spatial.distance import cdist




class ConnectivityMatrix:
    """
    Compute connectivity matrix between clusters
    """

    def __init__(self, clusters:List[Cluster],epsilon,limit_radian,n_dim,parallel):
        """Create instance of ConnectivityMatrix class. Extract all necessary data from clusters and into array

        :param clusters: List of Clusters
        :type clusters: List[Cluster]
        :param epsilon: epsilon
        :type epsilon: float
        :param limit_radian: limit radian distance between clusters
        :type limit_radian: int
        :param n_dim: number of dimensions
        :type n_dim: int
        :param parallel: use parallel computing via ray
        :type parallel: bool
        """

        self.data = []
        self.epsilon = epsilon
        self.limit_radian = limit_radian
        self.parallel = parallel
        for cluster in clusters:
            self.data.append([cluster.centroid.reshape(1,-1),#0
            len(cluster.galaxies),#1
            cluster.get_length(),#2
            cluster.rotated_cube,#3
            cluster.galaxies[:,:n_dim],
            cluster.isWasComplete()
            ])
        self.data = np.array(self.data,dtype=object)
    
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
        if self.parallel:
            data = ray.put(self.data)
            graph = ray.get([self.parallel_connectivity_matrix.remote(self, data, subset) for subset in self.split_array(arange(len(self.data)), 64)])
            graph = concatenate( graph, axis = 0 )
        else:
            graph = self.connectivity_matrix(self.data, arange(len(self.data)))

        graph = graph+transpose(graph)
        graph = graph-eye(len(self.data))
        return graph


    def connectivity_matrix(self, data , rows):
        """Utility method for connectivity matrix. Compute part of matrix

        :param data: data of clusters
        :type data: array
        :param rows: rows to compute
        :type rows: array
        :return: part of connectivity matrix
        :rtype: array
        """
        new_graph = zeros((len(rows), len(data)))
        for i in range(len(rows)):
            for j in range(rows[i], len(data)):

                if(rows[i] == j):
                    new_graph[i, j] = 1
                    continue

                halfsum_len = (data[rows[i],2]+data[j,2])/2
                centroid_i = data[rows[i],0]
                centroid_j = data[j,0]
                n_dim = len(centroid_i)
                cube_i = data[rows[i],3]
                cube_j = data[j,3]
                p_i = data[rows[i],4]
                p_j = data[j,4]
                
                if(data[rows[i],5] and data[j,5]):
                    continue
                
                if((data[rows[i],1]==1) and (data[j,1]==1)):
                    if(cdist(centroid_i,centroid_j,'euclidean')<self.epsilon):
                        new_graph[i, j] = 1
                    continue

                centroids_diff =centroid_i[0]-centroid_j[0]
                dim_level_check = True
                for k in range(n_dim):
                    if(halfsum_len<np.abs(centroids_diff[k])):
                        dim_level_check = False
                        continue
                if(not dim_level_check):
                    continue


                if(np.arccos(1- cdist(centroid_i,centroid_j,'cosine')) > self.limit_radian):
                    continue
 
                if(distance.cdist(p_i,p_j).min()>10*self.epsilon):
                    continue

                if(self.check_collision(cube_i,p_j)):
                    new_graph[i, j] = 1
                elif(self.check_collision(cube_j, p_i)):
                    new_graph[i, j] = 1
        return new_graph

    @ray.remote
    def parallel_connectivity_matrix(self, data, rows):
        """Utility method for connectivity matrix. Compute part of matrix in parallel using ray

        :param data: data of clusters
        :type data: array
        :param rows: rows to compute
        :type rows: array
        :return: part of connectivity matrix
        :rtype: array
        """
        return self.connectivity_matrix(data, rows)
        
    def check_collision(self, cube, p):
        """Check colision of regions of space of clusters

        :param cube: region of space of cluster in form of coordinates
        :type cube: array
        :param p: points to check
        :type p: array
        :return: if ``p`` have points inside ``cube``
        :rtype: bool
        """
        delaunay = Delaunay(cube)
        for gal in p:
            if(delaunay.find_simplex(gal) >= 0):
                return True
        return False