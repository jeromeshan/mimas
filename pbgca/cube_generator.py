from numpy import transpose, array, ndarray
import numpy as np

class CubeGenerator:
    """
    Generating n-dimensional cubes
    """

    @staticmethod
    def n_dim_cube(n, init_length = 1, init_width = 1):
        """Generating n-dimensional cube with given width and length

        :param n: number of dimensions
        :type n: int
        :param init_length: length of cube, defaults to 1
        :type init_length: float
        :param init_width: width of cube, defaults to 1
        :type init_width: float
        :return: coordinates of n-dim cube
        :rtype: array
        """
        unit = [[ -0.5, -0.5, 0.5, 0.5], [ -0.5, 0.5, 0.5, -0.5]]

        if n != 2:
            for k in range(n-2):
                l = len(unit[0])
                for i in range(len(unit)):
                    unit[i] = unit[i]+unit[i]
                unit.append( [-0.5]*l + [0.5]*l )
        
        for l in range(len(unit)):
            if l == 0:
                unit[l] = [element * init_length for element in unit[l]]
            else:
                unit[l] = [element * init_width for element in unit[l]]

        return transpose(array(unit))