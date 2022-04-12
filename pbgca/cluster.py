from scipy.spatial import Delaunay
from numpy import array, ndarray
import numpy as np
from .cube_generator import CubeGenerator

class Cluster:
    """ Class denife cluster for our clusterer.
    Cluster can be defined by: center of cluster, points within a cluster and boundaries of cluster's region of space represented as number of coordinates of n-dim cuboid

    Some example
    docstring::

        from pbgca import Cluster

        cluster = Cluster([1,1],epsilon = 2)

        cluster.get_length()# result 2

        cluster.grow()# cluster's region of space expands. Coordinates of cuboid expands

    Main methods here are :py:meth:`grow` and :py:meth:`__init__`
    """



    non_rotated_cube = None
    rotated_cube = None# shifted coor of rotated cube
    centroid = None# coor of center
    galaxies = None
    grow_limit = None
    new_galaxies_koef = None
    grow_function = None
    elongate_grow = None
    prev_n = None
    prev_v = None
    lr = None
    n_dim = 0
    times_grow = 0
    epsilon = 0

    # '''
    # center - coordinates of center of cluster (in case on sigle galaxy coordinates on galaxy)
    # non_rotated_cube - coordinates of non rotated parallelepiped that fits to cluster
    # galaxies - coordinates on galaxies
    # init_length - ration on longer part of parallelepiped to shorters
    # '''
    def __init__(self, center, non_rotated_cube = None,
                galaxies = None, epsilon = 0.5, n_dim = 3,
                grow_limit = 5, prev_n = None, prev_v = None,
                lr =1, elongate_grow = 1, grow_function = 'density'):
        """Create instance of cluster

        :param center: center of cluster
        :type center: array
        :param non_rotated_cube: coordinates of non rotated cube, defaults to None
        :type non_rotated_cube: array, optional
        :param galaxies: list of points in cluster, defaults to None
        :type galaxies: array, optional
        :param epsilon: epsilon, defaults to 0.5
        :type epsilon: float, optional
        :param n_dim: number of dimensions, defaults to 3
        :type n_dim: int, optional
        :param grow_limit: limit number of grow iterations, defaults to 5
        :type grow_limit: int, optional
        :param prev_n: max size of previous clusters that merge into current (by number of points), defaults to None
        :type prev_n: int, optional
        :param prev_v: density of max size of previous cluster that merge into current , defaults to None
        :type prev_v: float, optional
        :param lr: learning rate, defaults to 1
        :type lr: float, optional
        :param elongate_grow: elongation factor, defaults to 1
        :type elongate_grow: float, optional
        :param grow_function: type of grow function, defaults to 'density'
        :type grow_function: 'density' or 'normal', optional
        """
        
        self.was_complete = False
        
        self.n_dim = n_dim 
        self.elongate_grow = elongate_grow
        self.grow_limit = grow_limit
        self.prev_n = prev_n
        self.prev_v = prev_v
        self.lr = lr
        self.grow_function = grow_function

        self.non_rotated_cube = CubeGenerator.n_dim_cube(self.n_dim, epsilon, epsilon)
        
        if(isinstance(non_rotated_cube, ndarray)):
            self.non_rotated_cube = non_rotated_cube
        if(not isinstance(galaxies, ndarray)):
            self.galaxies = array([center])
        else:
            self.galaxies = array(galaxies)
        
        self.centroid = center[:self.n_dim]
        self.rotated_cube = self.rotate(self.centroid, self.non_rotated_cube)+self.centroid
        
        for gal in self.galaxies:
            if(Delaunay(self.rotated_cube).find_simplex(gal[:self.n_dim]) < 0):
                raise ValueError
        
    def get_length(self):
        """Get length of cluster

        :return: length of cluster
        :rtype: float
        """
        length  =  self.non_rotated_cube[2, 0] - self.non_rotated_cube[0, 0]
        return length


    def get_width(self):
        """Get width of cluster

        :return: width of cluster
        :rtype: float
        """
        width =  self.non_rotated_cube[1, 1] - self.non_rotated_cube[0, 0]
        return width

    def get_volume(self):
        """Get volume of cluster

        :return: volume of cluster
        :rtype: float
        """
        volume = self.get_length()
        for i in range(self.n_dim-1):
            volume*=self.get_width()
        return volume

        
    def rotate(self, vector, points):    
        """Rotate cluster to facing centain direction

        :param vector: direction vector
        :type vector: array
        :param points: points of cube
        :type points: array
        :return: rotated points of cube
        :rtype: array
        """
        if(not np.any(vector)):
            return points

        x= array([1]+[0]*(self.n_dim-1))[np.newaxis, :].T
        y= array(vector)[np.newaxis, :].T
        y = y/np.linalg.norm(y)

        u = x/np.linalg.norm(x)
        v = y-(u.T@y)*u
        v = v/np.linalg.norm(v)

        cost = (x.T@y/np.linalg.norm(x)/np.linalg.norm(y))[0][0]
        sint = np.sqrt(1-np.power(cost,2))

        R = np.eye(len(x)) - u.dot(u.T) - v.dot(v.T)+(np.concatenate([u,v],axis = 1)@np.array([[cost, -sint],[sint,cost]]))@np.concatenate([u,v],axis = 1).T
        
        rotmat = []
        for p in points:
            rotmat.append(R@p)

        return array(rotmat)

    def isComplete(self):
        """If cluster stops growth

        :return: if cluster stops growth
        :rtype: bool
        """
        return self.times_grow == self.grow_limit

    def isWasComplete(self):
        """If cluster stops growth more than step ago

        :return: if cluster stops growth more than step ago
        :rtype: bool
        """
        return self.was_complete


    def grow(self):
        """Grow of cluster
        """
        if(self.isComplete()):
            
            self.was_complete = True
            return
        self.times_grow += 1

        if(self.prev_n is None):
            composite_koef = np.power( self.lr  * (1 + 1/len(self.galaxies)),1/self.n_dim)
        else:
            if (self.grow_function == 'density'):
                nu1 = len(self.galaxies)*1.0/self.get_volume()
                nu2 = self.prev_n*1.0 / self.prev_v
                composite_koef = np.clip(np.power( self.lr*(len(self.galaxies)*1.0/self.prev_n) * (1 + nu2/nu1) ,1/self.n_dim), 1, 2)
            elif (self.grow_function == 'normal'):
                composite_koef = np.clip(np.power( self.lr*(len(self.galaxies)*1.0/self.prev_n) * (1 + 1/len(self.galaxies)) ,1/self.n_dim), 1, 2)
            else:
                raise ValueError('Incorrect grow_function value')
        self.prev_n = None
        self.prev_v = None
        self.non_rotated_cube[0] = self.non_rotated_cube[0]*composite_koef
        self.non_rotated_cube[1:,:] = self.non_rotated_cube[1:,:]*(1 + (composite_koef-1)/self.elongate_grow)        
        self.rotated_cube = self.rotate(self.centroid, self.non_rotated_cube)+self.centroid         