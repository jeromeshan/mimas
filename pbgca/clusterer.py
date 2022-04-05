from typing import List
import ray
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import cdist
import time
from numpy import transpose, array, arange, concatenate, eye, unique, dot, zeros, append, ones
import numpy as np
from numpy.linalg import norm
# from .cluster import Cluster, n_dim_cube
from cluster import Cluster, n_dim_cube

class ConnectivityMatrix:
    def __init__(self, clusters:List[Cluster],epsilon,limit_radian):

        self.data = []
        self.epsilon = epsilon
        self.limit_radian = limit_radian
        for cluster in clusters:
            self.data.append([cluster.centroid.reshape(1,-1),#0
            len(cluster.galaxies),#1
            cluster.get_length(),#2
            cluster.rotated_cube])#3
        self.data = np.array(self.data)
        # for el in self.data:
        #     print(el)
    
    def split_array(self, a, n):
        k, m = divmod(len(a), n)
        return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

    def get_matrix(self):
        data = ray.put(self.data)
        graph = ray.get([self.parallel_connectivity_matrix.remote(self, data, subset) for subset in self.split_array(arange(len(self.data)), 8)])
        graph = concatenate( graph, axis = 0 )
        # graph = self.connectivity_matrix(self.clusters, arange(len(self.clusters)))
        graph = graph+transpose(graph)
        graph = graph-eye(len(self.data))
        return graph

    @ray.remote
    def parallel_connectivity_matrix(self, data , rows):
        new_graph = zeros((len(rows), len(data)))
        for i in range(len(rows)):
            for j in range(rows[i], len(data)):

                halfsum_len = (data[rows[i],2]+data[j,2])/2
                centroid_i = data[rows[i],0]
                centroid_j = data[j,0]
                n_dim = len(centroid_i)
                cube_i = data[rows[i],3]
                cube_j = data[j,3]
                                
                centroids_diff =centroid_i[0]-centroid_j[0]
                dim_level_check = True
                for k in range(n_dim):
                    if(halfsum_len<np.abs(centroids_diff[k])):
                        dim_level_check = False
                        continue
                if(not dim_level_check):
                    continue

                if((data[rows[i],1]==1) and (data[rows[i],1]==1)):
                    if(cdist(centroid_i,centroid_j,'euclidean')<self.epsilon):
                        new_graph[i, j] = 1
                    continue

                # if(cdist(centroid_i,centroid_j,'euclidean') > halfsum_len):
                #     continue 

                if(np.arccos(1 - cdist(centroid_i,centroid_j,'cosine')) > self.limit_radian):
                    continue
 
                if(self.check_collision(cube_i,cube_j)):
                    new_graph[i, j] = 1
                elif(self.check_collision(cube_j, cube_i)):
                    new_graph[i, j] = 1
        return new_graph

    def check_collision(self, a, b):
        n = len(a[0])-2
        bases = []
        bases.append(a[1]-a[0])
        bases.append(a[3]-a[0])
        for i in range(n):
            bases.append(a[2**(i+2)]-a[0])
        for point in b:
            u = point-a[0]
            for v in bases:
                if not ((0 <= dot(u, v)) and (dot(u, v) <= dot(v, v))):
                    continue
                return True
            # if ( all((0 <= dot(u, v)) and (dot(u, v) <= dot(v, v)) for v in bases)):
            #     return True
        return False

    

class Clusterer:
    lr = 0

    def __init__(self, epsilon = 0.5, lr = 1, max_iter = 30, limit_radian = 0.01, grow_limit = 3):
        self.clusters = []
        self.epsilon = epsilon
        self.lr = lr
        self.max_iter = max_iter
        self.limit_radian = limit_radian
        self.grow_limit = grow_limit
 
    def merge(self):
        # clusters_id = ray.put(self.clusters)
        # graph = ray.get([self.parallel_connectivity_matrix.remote(self, clusters_id, subset) for subset in self.split_array(arange(len(self.clusters)), 8)])
        # graph = concatenate( graph, axis = 0 )
        # graph = self.connectivity_matrix(self.clusters, arange(len(self.clusters)))

        matrix_gen = ConnectivityMatrix(self.clusters,self.epsilon,self.limit_radian)
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

        cube = n_dim_cube(self.n_dim, length*2+self.epsilon/2, width*2+self.epsilon/2)
 
        return Cluster(center, cube, galaxies,  n_dim = self.n_dim, prev_n= len(max_prev_cluster.galaxies),prev_v=max_prev_cluster.get_volume(), grow_limit = self.grow_limit, lr = self.lr)
 
    def compress_cluster(self, cluster):
 
        galaxies = cluster.galaxies
        if(galaxies.ndim == 1 ):
            galaxies = array([galaxies])
        vertex_points = cluster.rotated_cube
 
        center = galaxies[:, :self.n_dim].mean(axis = 0)
 
        projections = center * dot(galaxies[:, :self.n_dim], transpose([center])) / dot(center, center)
 
        distances_on_line = norm(projections-center, axis = 1)
        vectors_from_line = galaxies[:, :self.n_dim]-projections

        length = distances_on_line.max()
        width = norm(vectors_from_line, axis = 1).max()
 
        cube = n_dim_cube(self.n_dim, length*2+self.epsilon/10, width*2+self.epsilon/10)
        
        return Cluster(center, cube, galaxies, n_dim = self.n_dim, lr = self.lr)
 
    def step(self):
 
        start = time.time()
        is_done = True
        for cluster in self.clusters:
            if(not cluster.isComplete()):
                is_done = False
                break
        print('\t','complete check time: ', time.time()-start, ' s')

        if (is_done):
            return False

        start = time.time()
        self.merge()
        print('\t','merge time: ', time.time()-start, ' s')

        start = time.time()
        for cluster in self.clusters:
            cluster.grow()
        print('\t','grow time: ', time.time()-start, ' s')

        return True

    def split_array(self, a, n):
        k, m = divmod(len(a), n)
        return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))
 

    def fit(self, data):
        try:
            ray.init()
            data_np = array(data)
            self.n_dim = len(data_np[0])
            self.clusters = [Cluster(append(data_np[i], i), epsilon = self.epsilon, n_dim = self.n_dim,lr = self.lr) for i in range(len(data_np)) ]
            iter_num = 1
            start = time.time()
            while(self.step()):
                print('iter : ', iter_num, ', n_clusters: ', len(self.clusters), ', time: ', time.time()-start, ' s')
                iter_num += 1
                start = time.time()

                if(iter_num > self.max_iter):
                    break

            galaxies = []

            galaxies = [append(self.clusters[i].galaxies, ones((len(self.clusters[i].galaxies), 1))*i , axis = 1) for i in range(len(self.clusters))]
            self.clusters = [self.compress_cluster(cluster) for cluster in self.clusters]
            galaxies = concatenate(galaxies)
            galaxies = galaxies[galaxies[:, self.n_dim].argsort()]  
            return galaxies[:, -1]
        except Exception as e:
            print(e)
        finally:
            ray.shutdown()