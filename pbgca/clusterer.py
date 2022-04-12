from typing import List
import ray
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist
import time
from numpy import transpose, array, arange, concatenate, eye, unique, dot, zeros, append, ones
import numpy as np
from numpy.linalg import norm
from .cluster import Cluster
from .cube_generator import CubeGenerator
from .connectivity_matrix import ConnectivityMatrix
from scipy.spatial import distance
    

class Clusterer:
    """Main clusterer class

    TODO: Short defenition of algorithm

    """

    def __init__(self, epsilon = 0.5, lr = 1, max_iter = 30,
                limit_radian = 0.01, grow_limit = 3, elongate_grow = 1,
                grow_function = 'density', min_diff = 1, parallel = False):
        """Create instance of clusterer

        :param epsilon: epsilon, defaults to 0.5
        :type epsilon: float, optional
        :param lr: learning rate, defaults to 1
        :type lr: int, optional
        :param max_iter: max number of interations, defaults to 30
        :type max_iter: int, optional
        :param limit_radian: limit radian distance between clusters, defaults to 0.01
        :type limit_radian: float, optional
        :param grow_limit: limit number of grow iterations, defaults to 3
        :type grow_limit: int, optional
        :param elongate_grow: elongation factor, defaults to 1
        :type elongate_grow: float, optional
        :param grow_function: type of grow function, defaults to 'density'
        :type grow_function: 'density' or 'normal', optional
        :param min_diff: min diff in number of clusters between steps, defaults to 1
        :type min_diff: int, optional
        :param parallel: use parallel computing via ray, defaults to False
        :type parallel: bool, optional
        """
        self.clusters = []
        self.epsilon = epsilon
        self.lr = lr
        self.max_iter = max_iter
        self.limit_radian = limit_radian
        self.grow_limit = grow_limit
        self.elongate_grow = elongate_grow
        self.grow_function = grow_function
        self.min_diff = min_diff
        self.parallel = parallel
 
    def merge(self):
        """
        Use ConnectivityMatrix to compute connectivity matrix and find connected components to merge them then
        """

        matrix_gen = ConnectivityMatrix(self.clusters,self.epsilon,self.limit_radian,self.n_dim,self.parallel)
        graph = matrix_gen.get_matrix()     

        graph = graph+transpose(graph)
        graph = graph-eye(len(self.clusters))
        graph = csr_matrix(graph)
        _, labels = connected_components(csgraph = graph, directed = False, return_labels = True)
        components = []
 
        np_clusters = array(self.clusters)
        components = [np_clusters[labels == label]  for label in unique(labels) ]
        new_clusters = list(map(self.collide_clusters, components))

        self.clusters = new_clusters        
 
    def collide_clusters(self, clusters:List[Cluster]):
        """Merge list of intersected clusters together

        :param clusters: List of clusters
        :type clusters: List[Cluster]
        :return: merged cluster
        :rtype: Cluster
        """
 
        if(len(clusters) == 1):
            return clusters[0]

        max_prev_cluster = max(clusters, key = lambda c: len(c.galaxies))
        galaxies = [cluster.galaxies for cluster in clusters]
        vertex_points = [cluster.rotated_cube for cluster in clusters]
 
        galaxies = concatenate( galaxies, axis = 0 )
        vertex_points = concatenate( vertex_points, axis = 0 )
 
        center = galaxies[:, :self.n_dim].mean(axis = 0)
 
        projections = center * dot(galaxies[:, :self.n_dim], transpose([center])) / dot(center, center)
 
        distances_on_line = norm(projections-center, axis = 1)
        vectors_from_line = galaxies[:, :self.n_dim]-projections
 
        length = distances_on_line.max()
        width = norm(vectors_from_line, axis = 1).max()

        cube = CubeGenerator.n_dim_cube(self.n_dim, length*2+self.epsilon/2, width*2+self.epsilon/2)
 
        return Cluster(center, cube, galaxies,  n_dim = self.n_dim, prev_n= len(max_prev_cluster.galaxies),prev_v=max_prev_cluster.get_volume(), grow_limit = self.grow_limit, lr = self.lr, elongate_grow = self.elongate_grow, grow_function=self.grow_function)
 
    def compress_cluster(self, cluster):
        """Compress clusters to minimal size. Calls in the last step of algoritm to level out grow effect

        :param cluster: cluster
        :type cluster: Cluster
        :return: Compressed cluster
        :rtype: Cluster
        """
        galaxies = cluster.galaxies
        if(galaxies.ndim == 1 ):
            galaxies = array([galaxies])
 
        center = galaxies[:, :self.n_dim].mean(axis = 0)
 
        projections = center * dot(galaxies[:, :self.n_dim], transpose([center])) / dot(center, center)
 
        distances_on_line = norm(projections-center, axis = 1)
        vectors_from_line = galaxies[:, :self.n_dim]-projections

        length = distances_on_line.max()
        width = norm(vectors_from_line, axis = 1).max()
 
        cube = CubeGenerator.n_dim_cube(self.n_dim, length*2+self.epsilon/10, width*2+self.epsilon/10)
        
        return Cluster(center, cube, galaxies, n_dim = self.n_dim)
 
    def step(self):
        """One step of algoritm. Grow and then merge all clusters

        :return: is last step
        :rtype: bool
        """
        # start = time.time()
        is_done = True
        for cluster in self.clusters:
            if(not cluster.isComplete()):
                is_done = False
                break
        # print('\t','complete check time: ', time.time()-start, ' s')

        if (is_done):
            return False
        
        
        if(not self.isFirstStep):
            # start = time.time()
            for cluster in self.clusters:
                cluster.grow()
            # print('\t','grow time: ', time.time()-start, ' s')
        else:
            self.isFirstStep = False

        # start = time.time()
        self.merge()
        # print('s\t','merge time: ', time.time()-start, ' s')

        return True
 

    def fit(self, data):
        """Fit method of algoritm. Launch all other steps

        :param data: data for clustering
        :type data: array
        :return: labels of points
        :rtype: array
        """
        if(self.parallel):
            if ray.is_initialized():
                ray.shutdown()
            ray.init()

        data_np = array(data)
        self.n_dim = len(data_np[0])
        self.clusters = [Cluster(append(data_np[i], i), epsilon = self.epsilon, n_dim = self.n_dim,lr = self.lr,elongate_grow = self.elongate_grow, grow_function=self.grow_function) for i in range(len(data_np)) ]
        iter_num = 1
        self.isFirstStep = True
        start = time.time()
        prev_clusters_number = len(self.clusters)
        while(self.step()):
            print('iter : ', iter_num, ', n_clusters: ', len(self.clusters), ', time: ', time.time()-start, ' s')
            iter_num += 1
            start = time.time()

            if ((self.min_diff>0) and (prev_clusters_number - len(self.clusters) <= self.min_diff)):
                break
            else:
                prev_clusters_number = len(self.clusters)
            
            if(iter_num > self.max_iter):
                break

        galaxies = []
        galaxies = [append(self.clusters[i].galaxies, ones((len(self.clusters[i].galaxies), 1))*i , axis = 1) for i in range(len(self.clusters))]
        self.clusters = [self.compress_cluster(cluster) for cluster in self.clusters]
        galaxies = concatenate(galaxies)
        galaxies = galaxies[galaxies[:, self.n_dim].argsort()]  

        if(self.parallel):
            ray.shutdown()
        
        self.labels_ = galaxies[:, -1]

        return self.labels_