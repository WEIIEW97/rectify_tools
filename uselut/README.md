Using Look-Up Table to Rectify Stereo Images
======================

A transfer from MATLAB version to c++ version.
----------------------

- [Using Look-Up Table to Rectify Stereo Images](#using-look-up-table-to-rectify-stereo-images)
  - [A transfer from MATLAB version to c++ version.](#a-transfer-from-matlab-version-to-c-version)
    - [File Tree](#file-tree)
    - [3rd-party](#3rd-party)
    - [Structure Interpretation](#structure-interpretation)
      - [utils.cpp](#utilscpp)
      - [lutParser.cpp](#lutparsercpp)
      - [rectImg.cpp](#rectimgcpp)

### File Tree

```
project
│   README.md
│   CMakeLists.txt    
│
└───src
│   │   lutParser.cpp
│   │   lutParser.h
│   │   rectImg.cpp
|   |   rectimg.h
│   |   useLut.cpp
|   |   useLut.h
|   |   utils.cpp
|   |   utils.h
└───data
    │   ${input_folder}
    │   ...
```

### 3rd-party
```cpp
#include <algorithm>
#include <bitset>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <tuple>
#include <unordered_set>
#include <vector>

#include <Eigen/Dense>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <time.h>
```

### Structure Interpretation
#### utils.cpp
- *normc* : Col-wise normalization.
  $$
  X_{i} = \frac{X_{i} - \min_{k, k\in i}(X)}{\max_{k, k\in i}(X) - \min_{k, k\in i}(X)}
  $$
- *meshgrid* : Return coordinate matrices from coordinate vectors.
Make N-D coordinate arrays for vectorized evaluations of N-D scalar/vector fields over N-D grids, given one-dimensional coordinate arrays x1, x2,…, xn.
- *dec2bin* : Convert decimal data to binary data(type: string).
- *bin2dec* : Convert binary data to decimal.
- *double2fixed* : Convert double point data to fixed point data.
- *fixed2double* : Convert fixed point data to double data.
- *sub2ind* : Convert subscripts to linear indices. 
- *ind2sub* : Convert linear indices to subscripts.
- *ismember* : Return true if element is in a set.

#### lutParser.cpp
- *lut_parser* : Parse look-up table and restore back it correspondingly and accordingly by LUT indices order. Then apply technique of bi-linear interpolation to interpolate all pixels.
- *sparse2dense*: Bi-linear interpolation. Expanding sparse martix to dense matrix.


#### rectImg.cpp
- *rect_img* : Find the right position for pixels by comparison and searching pairwisely between original matrix and rectified matrix.
- *bilinear_remap* : Same to bi-linear interpolation.