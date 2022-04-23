#!python
# cython: embedsignature=True, binding=True
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist

import numpy as np
cimport numpy as np
np.import_array()

DTYPE = np.double

ctypedef np.double_t DTYPE_t

cimport cython
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False) 
def connectivity_matrix_cython(data, Py_ssize_t [:] rows,float epsilon, float limit_radian):

    cdef np.ndarray[DTYPE_t, ndim=2] new_graph = np.zeros((len(rows), len(data)), dtype=DTYPE)
    cdef DTYPE_t halfsum_len
    cdef np.ndarray[DTYPE_t, ndim=2] centroid_i
    cdef np.ndarray[DTYPE_t, ndim=2] centroid_j
    cdef DTYPE_t n_dim
    cdef np.ndarray[DTYPE_t, ndim=2] cube_i
    cdef np.ndarray[DTYPE_t, ndim=2] cube_j 
    cdef np.ndarray[DTYPE_t, ndim=2] p_i 
    cdef np.ndarray[DTYPE_t, ndim=2] p_j 
    cdef np.ndarray[DTYPE_t, ndim=1] centroids_diff
    cdef bint dim_level_check
    
    for i in range(len(rows)):
        for j in range(rows[i], len(data)):

            if(rows[i] == j):
                new_graph[i, j] = 1
                continue

            centroid_i = data[rows[i],0]
            centroid_j = data[j,0]

            if(data[rows[i],5] and data[j,5]):
                continue

            if((data[rows[i],1]==1) and (data[j,1]==1)):
                if(cdist(centroid_i,centroid_j,'euclidean')<epsilon):
                    new_graph[i, j] = 1
                continue

            centroids_diff =centroid_i[0]-centroid_j[0]
            dim_level_check = True
            halfsum_len = (data[rows[i],2]+data[j,2])/2
            n_dim = len(centroid_i)

            for k in range(int(n_dim)):
                if(halfsum_len<np.abs(centroids_diff[k])):
                    dim_level_check = False
                    continue
            if(not dim_level_check):
                continue


            if(np.arccos(1- cdist(centroid_i,centroid_j,'cosine')) > limit_radian):
                continue
                
            p_i = data[rows[i],4]
            p_j = data[j,4]
            
            if(cdist(p_i,p_j).min()>10*epsilon):
                continue

            cube_i = data[rows[i],3]
            cube_j = data[j,3]
            
            if(check_collision(cube_i,p_j)):
                new_graph[i, j] = 1
            elif(check_collision(cube_j, p_i)):
                new_graph[i, j] = 1
    return new_graph


def check_collision(cube, p):

    delaunay = Delaunay(cube)
    for gal in p:
        if(delaunay.find_simplex(gal) >= 0):
            return True
    return False