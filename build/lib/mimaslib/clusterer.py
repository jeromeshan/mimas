# import numpy as np
from typing import List
import ray
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import time
from numpy import transpose,array,arange,concatenate,eye,unique,dot,zeros,append,ones
import numpy as np
from numpy.linalg import norm
from .cluster import Cluster
# from fancyplot import *

class Clusterer:
 
    unit_cube=(transpose(array([[0, 0, 1, 1, 0, 0, 1, 1],[0, 1, 1, 0, 0, 1, 1, 0],[0, 0, 0, 0, 1, 1, 1, 1]]))-0.5)
    init_cluster_length = None
    init_cluster_width = None
    coefs = 0
 
    '''
        init_length - ration on longer part of parallelepiped to shorters
        coefs - array of coefs to steps of algo
    '''
    def __init__(self,init_cluster_length=0.5,init_cluster_width=0.05, coefs = 1+1/arange(1,100)[15:]*5, max_iter = 30, limit_radian = 0.01):
        self.clusters=[]
        self.init_cluster_length = init_cluster_length
        self.init_cluster_width = init_cluster_width
        self.coefs = coefs
        self.max_iter = max_iter
        self.limit_radian = limit_radian
 
    def merge(self):
 
        clusters_len=len(self.clusters)
 
 
        clusters_id = ray.put(self.clusters)
 
        graph=ray.get([self.parallel_connectivity_matrix.remote(self,clusters_id, subset) for subset in self.split_array(arange(len(self.clusters)),4)])
        graph = concatenate( graph, axis=0 )

        # graph = self.connectivity_matrix(self.clusters, arange(len(self.clusters)))

        graph = graph+transpose(graph)
        graph=graph-eye(len(self.clusters))
 
        graph = csr_matrix(graph)
        n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
 
        components=[]
 
        np_clusters=array(self.clusters)
 
        components=[np_clusters[labels==label]  for label in unique(labels) ]
 
        new_clusters=list(map(self.collide_clusters,components))
 
        self.clusters=new_clusters        
 
 
    def check_collision(self,a:Cluster, b: Cluster):      
        i=a.rotated_cube[1]-a.rotated_cube[0]
        j=a.rotated_cube[3]-a.rotated_cube[0]
        k=a.rotated_cube[4]-a.rotated_cube[0]
        for point in b.rotated_cube:
            u=point-a.rotated_cube[0]
            u_i,u_j,u_k=dot(u,i),dot(u,j),dot(u,k)
            if((0<= u_i)and(0<= u_j)and(0<= u_k)and
               (u_i <= dot(i,i))and(u_j <= dot(j,j))and(u_k <= dot(k,k))):
                return True
 
        i=b.rotated_cube[1]-b.rotated_cube[0]
        j=b.rotated_cube[3]-b.rotated_cube[0]
        k=b.rotated_cube[4]-b.rotated_cube[0]
        for point in a.rotated_cube:
            u=point-b.rotated_cube[0]
            u_i,u_j,u_k=dot(u,i),dot(u,j),dot(u,k)
            if((0<= u_i)and(0<= u_j)and(0<= u_k)and
               (u_i <= dot(i,i))and(u_j <= dot(j,j))and(u_k <= dot(k,k))):
                return True
        return False
 
    def collide_clusters(self, clusters:List[Cluster]):
 
        if(len(clusters)==1):
            return clusters[0]
        galaxies=[cluster.galaxies for cluster in clusters]
        vertex_points=[cluster.rotated_cube for cluster in clusters]
 
        galaxies=concatenate( galaxies, axis=0 )
        vertex_points=concatenate( vertex_points, axis=0 )
 
        center = galaxies[:,:3].mean(axis=0)
 
        projections = center * dot(galaxies[:,:3], transpose([center])) / dot(center, center)
 
        distances_on_line=norm(projections-center,axis=1)
        vectors_from_line=galaxies[:,:3]-projections
 
        length = distances_on_line.max()
        width=norm(vectors_from_line,axis=1).max()
 
        cube=self.unit_cube.copy()*[width*2+self.init_cluster_width,width*2+self.init_cluster_width,length*2+self.init_cluster_length]
 
        return Cluster(center,cube,galaxies,self.init_cluster_length)
 
    def compress_cluster(self,cluster):
 
        galaxies=cluster.galaxies
        if(galaxies.ndim==1 ):
            galaxies=array([galaxies])
        vertex_points=cluster.rotated_cube
 
        center = galaxies[:,:3].mean(axis=0)
 
        projections = center * dot(galaxies[:,:3], transpose([center])) / dot(center, center)
 
        distances_on_line=norm(projections-center,axis=1)
        vectors_from_line=galaxies[:,:3]-projections
 
        length = distances_on_line.max()
        width=norm(vectors_from_line,axis=1).max()
 
        cube=self.unit_cube.copy()*[width*2+self.init_cluster_width/10,width*2+self.init_cluster_width/10,length*2+self.init_cluster_length/10]
 
        return Cluster(center,cube,galaxies,self.init_cluster_length)
 
    def step(self,grow_coef):
 
        is_done=True
        for cluster in self.clusters:
            if(not cluster.is_complete):
                is_done=False
                break
 
        if (is_done):
            return False
 
        for cluster in self.clusters:
            cluster.grow(1+(grow_coef-1)/ (1 + cluster.times_grow/5 + len(cluster.galaxies)/10))
        self.merge()
        return True
 
    def split_array(self,a, n):
        k, m = divmod(len(a), n)
        return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))
 
    @ray.remote
    def parallel_connectivity_matrix(self,clusters ,rows):
        new_graph=zeros((len(rows),len(clusters)))
        for i in range(len(rows)):
            for j in range(rows[i],len(clusters)):
                cluster_j=clusters[j]
                cluster_i=clusters[rows[i]]
                if(norm(cluster_i.centroid-cluster_j.centroid)>(cluster_i.get_length()+cluster_j.get_length())/2):
                    continue 
 
                if(np.arccos((dot(cluster_i.centroid, cluster_j.centroid) / (norm(cluster_i.centroid) * norm(cluster_j.centroid))))>self.limit_radian):
                    continue
 
                if(self.check_collision(cluster_i,cluster_j)):
                    new_graph[i,j]=1
        return new_graph
 
    def connectivity_matrix(self,clusters ,rows):
        new_graph=zeros((len(rows),len(clusters)))
        for i in range(len(rows)):
            for j in range(rows[i],len(clusters)):
                cluster_j=clusters[j]
                cluster_i=clusters[rows[i]]
                if(norm(cluster_i.centroid-cluster_j.centroid)>(cluster_i.get_length()+cluster_j.get_length())):
                    continue 
 
                if(np.arccos((dot(cluster_i.centroid, cluster_j.centroid) / (norm(cluster_i.centroid) * norm(cluster_j.centroid))))>self.limit_radian):
                    continue
 
                if(self.check_collision(cluster_i,cluster_j)):
                    new_graph[i,j]=1
        return new_graph
    def fit(self,data):
        try:
            ray.init()
            data_np=array(data)
            self.clusters=[Cluster(append(data_np[i],i), init_length = self.init_cluster_length,  init_width = self.init_cluster_width) for i in range(len(data_np)) ]
            iter_num=1
            start = time.time()
            while(self.step(self.coefs[iter_num-1])):
                print('iter : ',iter_num,', n_clusters: ', len(self.clusters),', time: ',time.time()-start,' s')
                iter_num+=1
                start = time.time()
    
                if(iter_num > self.max_iter):
                    break
    
            galaxies=[]

            galaxies=[append(self.clusters[i].galaxies, ones((len(self.clusters[i].galaxies),1))*i , axis=1) for i in range(len(self.clusters))]
            self.clusters=[self.compress_cluster(cluster) for cluster in self.clusters]
            galaxies=concatenate(galaxies)
            galaxies = galaxies[galaxies[:,3].argsort()]  
            return galaxies[:,4]
        except Exception as e:
            print(e)
        finally:
            ray.shutdown()
