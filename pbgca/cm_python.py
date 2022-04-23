from scipy.spatial import Delaunay
from numpy import  zeros
import numpy as np
from scipy.spatial.distance import cdist

def connectivity_matrix_python(data , rows,epsilon,limit_radian):

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
                if(cdist(centroid_i,centroid_j,'euclidean')<epsilon):
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


            if(np.arccos(1- cdist(centroid_i,centroid_j,'cosine')) > limit_radian):
                continue

            if(cdist(p_i,p_j).min()>10*epsilon):
                continue

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