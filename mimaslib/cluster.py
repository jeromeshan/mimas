from scipy.spatial import Delaunay
import vg
from pytransform3d.rotations import matrix_from_axis_angle
from scipy.spatial.transform import Rotation as Rot
from numpy import transpose, array, ndarray, hstack


class Cluster():
    # non shifted shape, easy to expand
    non_rotated_cube= None
    rotated_cube=None# shifted coor of rotated cube
    centroid=None# coor of center
    galaxies=None

    is_complete=False
    times_grow=0
    '''
    center - coordinates of center of cluster (in case on sigle galaxy coordinates on galaxy)
    non_rotated_cube - coordinates of non rotated parallelepiped that fits to cluster
    galaxies - coordinates on galaxies
    init_length - ration on longer part of parallelepiped to shorters
    '''
    def __init__(self, center,non_rotated_cube=None,galaxies=None, init_length=10):
        
        tmp_l= [0, 0, 0, 0, 1, 1, 1, 1]
        self.non_rotated_cube = (transpose(array([[0, 0, 1, 1, 0, 0, 1, 1],[0, 1, 1, 0, 0, 1, 1, 0],
                                                  [element * init_length for element in tmp_l]]))-array([0.5,0.5,0.5*init_length]))*0.05
        
        if(isinstance(non_rotated_cube, ndarray)):
            self.non_rotated_cube=non_rotated_cube
        if(not isinstance(galaxies, ndarray)):
            self.galaxies=array([center])
        else:
            self.galaxies=array(galaxies)
        
        self.centroid=center[:3]
        self.rotated_cube = self.rotate(self.centroid,self.non_rotated_cube)+self.centroid
        
        for gal in self.galaxies:
            if(Delaunay(self.rotated_cube).find_simplex(gal[:3]) < 0):
                raise ValueError
        
        
    def rotate(self, vector, points):    
        vector = vg.normalize(vector)
        axis = vg.perpendicular(vg.basis.z, vector)
        angle = vg.angle(vg.basis.z, vector, units='rad')
        
        a = hstack((axis, (angle,)))
        R = matrix_from_axis_angle(a)
        
        r = Rot.from_matrix(R)
        rotmat = r.apply(points)
        
        return rotmat

    def grow(self,coef):
        if(self.times_grow==5):
            self.is_complete=True
            return
        self.times_grow+=1
        self.non_rotated_cube=self.non_rotated_cube*coef
        self.rotated_cube = self.rotate(self.centroid,self.non_rotated_cube)+self.centroid         
    