# mimas - clusterer

Agglomenative clustering algorithm for galaxy formations considering physical form of galaxy's structures

Input:
  3d coordinates of galaxies in form of np matrix or pandas dataframe (units: Mpc)
  Example:

| x      | y | z    |
| ------------- | ------------- |------------- |
| 1      | 34       | 0.45   |
| -1   | 94        | -0.322      |

Ouput:
  cluster labels (galaxies without cluster have label -1) in form of .np
  Optional: html plot (via plotly.dash)
