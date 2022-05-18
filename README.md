Rectify_tools README
===================

This project is using look-up-table to rectify images.
- [Rectify_tools README](#rectify_tools-readme)
  - [Basic Knowledge](#basic-knowledge)
    - [Optical Flow](#optical-flow)
    - [Bilinear Interpolation](#bilinear-interpolation)
  - [Structure in MATLAB](#structure-in-matlab)
  - [LUTParser](#lutparser)
    - [Loading LUT file](#loading-lut-file)
    - [Parsing LUT file](#parsing-lut-file)
  - [Rectify image by LUT](#rectify-image-by-lut)
    - [Algorithm Brief](#algorithm-brief)

## Basic Knowledge
### Optical Flow
**Definition**

For a (2D + t)-dimensional case (3D or n-D cases are similar) a voxel at location $\displaystyle (x,y,t)$ with intensity $\displaystyle I(x,y,t)$ will have moved by $\displaystyle \Delta x$, $\displaystyle \Delta y$ and $\displaystyle \Delta t$ between the two image frames, and the following brightness constancy constraint can be given:

$$I(x,y,t) = I(x+\Delta x, y + \Delta y, t + \Delta t)$$
Assuming the movement to be small, the image constraint at ${\displaystyle I(x,y,t)}$ with Taylor series can be developed to get:

$$\begin{aligned}
{\displaystyle I(x+\Delta x,y+\Delta y,t+\Delta t)=I(x,y,t)+{\frac {\partial I}{\partial x}}\,\Delta x+{\frac {\partial I}{\partial y}}\,\Delta y+{\frac {\partial I}{\partial t}}\,\Delta t+{\delta}}
\end{aligned}$$

By truncating the higher order terms (which performs a linearization) it follows that:
$$
{\displaystyle {\frac {\partial I}{\partial x}}\Delta x+{\frac {\partial I}{\partial y}}\Delta y+{\frac {\partial I}{\partial t}}\Delta t=0}
$$
or, dividing by ${\displaystyle \Delta t}$,

$$
{\displaystyle {\frac {\partial I}{\partial x}}{\frac {\Delta x}{\Delta t}}+{\frac {\partial I}{\partial y}}{\frac {\Delta y}{\Delta t}}+{\frac {\partial I}{\partial t}}{\frac {\Delta t}{\Delta t}}=0}
$$
which results in
$$
{\displaystyle {\frac {\partial I}{\partial x}}V_{x}+{\frac {\partial I}{\partial y}}V_{y}+{\frac {\partial I}{\partial t}}=0}
$$

where ${\displaystyle V_{x},V_{y}}$ are the ${\displaystyle x}$ and ${\displaystyle y}$ components of the velocity or optical flow of ${\displaystyle I(x,y,t)}$ and $\tfrac{\partial I}{\partial x}$,$\tfrac{\partial I}{\partial y}$, $\tfrac{\partial I}{\partial t}$  are the derivatives of the image at ${\displaystyle (x,y,t)}$ in the corresponding directions. ${\displaystyle I_{x}}I_{x},{\displaystyle I_{y}}$  and ${\displaystyle I_{t}}$  can be written for the derivatives in the following.

Thus:

$$
{\displaystyle I_{x}V_{x}+I_{y}V_{y}=-I_{t}}
$$
or

$$
{\displaystyle \nabla I\cdot {\vec {V}}=-I_{t}}
$$


### Bilinear Interpolation
**Introduction**

In mathematics, bilinear interpolation is a method for interpolating functions of two variables (e.g., x and y) using repeated linear interpolation. It is usually applied to functions sampled on a 2D rectilinear grid, though it can be generalized to functions defined on the vertices of (a mesh of) arbitrary convex quadrilaterals.

Bilinear interpolation is performed using linear interpolation first in one direction, and then again in the other direction. Although each step is linear in the sampled values and in the position, the interpolation as a whole is not linear but rather quadratic in the sample location.

Bilinear interpolation is one of the basic resampling techniques in computer vision and image processing, where it is also called bilinear filtering or bilinear texture mapping.

**Computation**

Suppose that we want to find the value of the unknown function f at the point (x, y). It is assumed that we know the value of f at the four points $Q_{11} = (x_1, y_1), Q_{12} = (x_1, y_2), Q_{21} = (x_2, y_1), Q_{22} = (x_2, y_2)$.

***Repeated linear interpolation***

We first do linear interpolation in the x-direction. This yields
$$
\begin{aligned}
&f\left(x, y_{1}\right)=\frac{x_{2}-x}{x_{2}-x_{1}} f\left(Q_{11}\right)+\frac{x-x_{1}}{x_{2}-x_{1}} f\left(Q_{21}\right) \\
&f\left(x, y_{2}\right)=\frac{x_{2}-x}{x_{2}-x_{1}} f\left(Q_{12}\right)+\frac{x-x_{1}}{x_{2}-x_{1}} f\left(Q_{22}\right)
\end{aligned}
$$
We proceed by interpolating in the y-direction to obtain the desired estimate:
$$
\begin{aligned}
f(x, y) &=\frac{y_{2}-y}{y_{2}-y_{1}} f\left(x, y_{1}\right)+\frac{y-y_{1}}{y_{2}-y_{1}} f\left(x, y_{2}\right) \\
&=\frac{y_{2}-y}{y_{2}-y_{1}}\left(\frac{x_{2}-x}{x_{2}-x_{1}} f\left(Q_{11}\right)+\frac{x-x_{1}}{x_{2}-x_{1}} f\left(Q_{21}\right)\right)+\frac{y-y_{1}}{y_{2}-y_{1}}\left(\frac{x_{2}-x}{x_{2}-x_{1}} f\left(Q_{12}\right)+\frac{x-x_{1}}{x_{2}-x_{1}} f\left(Q_{22}\right)\right) \\
&=\frac{1}{\left(x_{2}-x_{1}\right)\left(y_{2}-y_{1}\right)}\left(f\left(Q_{11}\right)\left(x_{2}-x\right)\left(y_{2}-y\right)+f\left(Q_{21}\right)\left(x-x_{1}\right)\left(y_{2}-y\right)+f\left(Q_{12}\right)\left(x_{2}-x\right)\left(y-y_{1}\right)+f\left(Q_{22}\right)\left(x-x_{1}\right)\left(y-y_{1}\right)\right) \\
&=\frac{1}{\left(x_{2}-x_{1}\right)\left(y_{2}-y_{1}\right)}\left[x_{2}-x \quad x-x_{1}\right]\left[\begin{array}{ll}
f\left(Q_{11}\right) & f\left(Q_{12}\right) \\
f\left(Q_{21}\right) & f\left(Q_{22}\right)
\end{array}\right]\left[\begin{array}{l}
y_{2}-y \\
y-y_{1}
\end{array}\right]
\end{aligned}
$$

Note that we will arrive at the same result if the interpolation is done first along the y direction and then along the x direction.

***Polynomial fit***

An alternative way is to write the solution to the interpolation problem as a multilinear polynomial
$$
{\displaystyle f(x,y)\approx a_{00}+a_{10}x+a_{01}y+a_{11}xy,}
$$
where the coefficients are found by solving the linear system
$$
\left[\begin{array}{llll}
1 & x_{1} & y_{1} & x_{1} y_{1} \\
1 & x_{1} & y_{2} & x_{1} y_{2} \\
1 & x_{2} & y_{1} & x_{2} y_{1} \\
1 & x_{2} & y_{2} & x_{2} y_{2}
\end{array}\right]\left[\begin{array}{l}
a_{00} \\
a_{10} \\
a_{01} \\
a_{11}
\end{array}\right]=\left[\begin{array}{c}
f\left(Q_{11}\right) \\
f\left(Q_{12}\right) \\
f\left(Q_{21}\right) \\
f\left(Q_{22}\right)
\end{array}\right]
$$
yielding the result
$$
\left[\begin{array}{l}
a_{00} \\
a_{10} \\
a_{01} \\
a_{11}
\end{array}\right]=\frac{1}{\left(x_{2}-x_{1}\right)\left(y_{2}-y_{1}\right)}\left[\begin{array}{cccc}
x_{2} y_{2} & -x_{2} y_{1} & -x_{1} y_{2} & x_{1} y_{1} \\
-y_{2} & y_{1} & y_{2} & -y_{1} \\
-x_{2} & x_{2} & x_{1} & -x_{1} \\
1 & -1 & -1 & 1
\end{array}\right]\left[\begin{array}{l}
f\left(Q_{11}\right) \\
f\left(Q_{12}\right) \\
f\left(Q_{21}\right) \\
f\left(Q_{22}\right)
\end{array}\right]
$$

***Weighted mean***

The solution can also be written as a weighted mean of the $f(Q)$ :
$$
f(x, y) \approx w_{11} f\left(Q_{11}\right)+w_{12} f\left(Q_{12}\right)+w_{21} f\left(Q_{21}\right)+w_{22} f\left(Q_{22}\right),
$$
where the weights sum to 1 and satisfy the transposed linear system
$$
\left[\begin{array}{cccc}
1 & 1 & 1 & 1 \\
x_{1} & x_{1} & x_{2} & x_{2} \\
y_{1} & y_{2} & y_{1} & y_{2} \\
x_{1} y_{1} & x_{1} y_{2} & x_{2} y_{1} & x_{2} y_{2}
\end{array}\right]\left[\begin{array}{c}
w_{11} \\
w_{12} \\
w_{21} \\
w_{22}
\end{array}\right]=\left[\begin{array}{c}
1 \\
x \\
y \\
x y
\end{array}\right]
$$

which simplifies to
$$
\begin{aligned}
&w_{11}=\left(x_{2}-x\right)\left(y_{2}-y\right) /\left(x_{2}-x_{1}\right)\left(y_{2}-y_{1}\right) \\
&w_{12}=\left(x_{2}-x\right)\left(y-y_{1}\right) /\left(x_{2}-x_{1}\right)\left(y_{2}-y_{1}\right) \\
&w_{21}=\left(x-x_{1}\right)\left(y_{2}-y\right) /\left(x_{2}-x_{1}\right)\left(y_{2}-y_{1}\right) \\
&w_{22}=\left(x-x_{1}\right)\left(y-y_{1}\right) /\left(x_{2}-x_{1}\right)\left(y_{2}-y_{1}\right)
\end{aligned}
$$

in agreement with the result obtained by repeated linear interpolation. The set of weights can also be interpreted as a set of generalized barycentric coordinates for a rectangle.

***Alternative matrix form***

Combining the above, we have
$$
f(x, y) \approx \frac{1}{\left(x_{2}-x_{1}\right)\left(y_{2}-y_{1}\right)}\left[f\left(Q_{11}\right) \quad f\left(Q_{12}\right) \quad f\left(Q_{21}\right) \quad f\left(Q_{22}\right)\right]\left[\begin{array}{cccc}
x_{2} y_{2} & -y_{2} & -x_{2} & 1 \\
-x_{2} y_{1} & y_{1} & x_{2} & -1 \\
-x_{1} y_{2} & y_{2} & x_{1} & -1 \\
x_{1} y_{1} & -y_{1} & -x_{1} & 1
\end{array}\right]\left[\begin{array}{c}
1 \\
x \\
y \\
x y
\end{array}\right]
$$

***On the unit square***

If we choose a coordinate system in which the four points where f is known are (0, 0), (1, 0), (0, 1), and (1, 1), then the interpolation formula simplifies to
$$
{\displaystyle f(x,y)\approx f(0,0)(1-x)(1-y)+f(1,0)x(1-y)+f(0,1)(1-x)y+f(1,1)xy,}
$$
or equivalently, in matrix operations:
$$
{\displaystyle f(x,y)\approx {\begin{bmatrix}1-x&x\end{bmatrix}}{\begin{bmatrix}f(0,0)&f(0,1)\\f(1,0)&f(1,1)\end{bmatrix}}{\begin{bmatrix}1-y\\y\end{bmatrix}}.}
$$

Here we also recognize the weights:

$$
{\displaystyle {\begin{aligned}w_{11}&=(1-x)(1-y),\\w_{12}&=(1-x)y,\\w_{21}&=x(1-y),\\w_{22}&=xy.\end{aligned}}}
$$
Alternatively, the interpolant on the unit square can be written as
$$
{\displaystyle f(x,y)=a_{00}+a_{10}x+a_{01}y+a_{11}xy,}
$$
where
$$
{\displaystyle {\begin{aligned}a_{00}&=f(0,0),\\a_{10}&=f(1,0)-f(0,0),\\a_{01}&=f(0,1)-f(0,0),\\a_{11}&=f(1,1)-f(1,0)-f(0,1)+f(0,0).\end{aligned}}}
$$

In both cases, the number of constants (four) correspond to the number of data points where $f$ is given.

## Structure in MATLAB

* *LUTParser(lut_file, int_len, frac_len)*  --> Raw2RectDenseMapX, Raw2RectDenseMapY, Rect2RawDenseMapX, Rect2RawDenseMapY

* *RectImgByLut(Raw2RectDenseMapX, Raw2RectDenseMapY, Rect2RawDenseMapX, Rect2RawDenseMapY, image, imageOrig)* --> imgRect
    * Image type of `imageOrig`, `image` is YUV, Yx3 channels (or U/Vx3 channels)


## LUTParser

### Loading LUT file 
Storing order in LUT: 
* `[1:3]` cpu_info, image cols, image rows
* `[4:7]` rect2raw_sample_rows, rect2raw_sample_cols, raw2rect_sample_rows, raw2rect_sample_cols
* `[8:11]` crop_start_x, crop_start_y, crop_end_x, crop_end_y
* `[15:i]` raw2rect_sample_x
  * raw2rect_sample_x = LUT[indices] / 2.0
  * i = 15+raw2rect_sample_col_num-1
* `[i+1:i+1+j]` raw2rect_sample_y
  * raw2rect_sample_y = LUT[indices] / 2.0
  * j = 15+raw2rect_sample_col_num+raw2rect_sample_row_num-1
* `[j+1:j+1+k]` raw2rect_delta_sample
  * k = raw2rect_sample_row_num * raw2rect_sample_col_num - 1
* `[k+1:k+1+h]` rect2raw_delta_sample
  * h = rect2raw_sample_row_num * rect2raw_sample_col_num - 1


### Parsing LUT file
By parsing the information stored in the LUT, the corresponding raw2rect/rect2raw comparison relationship is obtained
* Get rec2rawSampleX/rec2rawSampleY, raw2recSampleX/raw2recSampleY from LUT according to index
* Get raw2rect_delta_sample_x/raw2rect_delta_sample_y, rect2raw_delta_sample_x/rect2raw_delta_sample_y from LUT according to index
* Accoring to$I_{n}(x_n,y_n,t_n)=I_{n-1}(x_{n-1},y_{n-1},t_{n-1})+\Delta I$ 
  * raw2rectMap = raw2rectSample - raw2rect_delta_sample
  * rect2rawMap = rect2rawSample - rect2raw_delta_sample
* Convert raw2rectMap/rect2rawMap from sparse martix to dense matrix
  * At this time, Bilinear Interpolation is used to do coordinate interpolation
  * dense_martix = Sparse2Dense(sparse_matrix)



## Rectify image by LUT 
Pass in {Raw2RectDenseMapX, Raw2RectDenseMapY, Rect2RawDenseMapX, Rect2RawDenseMapY} and the original Yx3C image generated by *LUTParser* to produce rectified image components

### Algorithm Brief
* Initialize a uint8 matrix of the same size as orig_img
* Splicing the transpose of xOrig2Rect/yOrig2Rect together through hconcat, traverse each point in each line (assuming scale=1 at this time)
* If there is a crop operation, then only the pixels in the crop area are operated
* Return to the pixel in the crop area to find the value in the original raw2rect and write it
* Pixel weight filling with Bilinear Interpolation
* get rectfied image