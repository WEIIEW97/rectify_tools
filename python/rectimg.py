import numpy as np
import math
import numba
import cv2
import json


def normc(x):
    if np.mean(x) >= 0:
        _norm = x / x.max(axis=0)
    else:
        _norm = (x - x.min(axis=0)) / x.ptp(axis=0)
    return _norm


def get_res_and_mode(model, *args):
    """parses input arguments.
    """
    res = np.array((model['width'], model['length']))
    mode = 0
    if len(args) <= 0:
        return

    istart = 0
    if isinstance(args[0], (int, float)):
        res = args[0]
        istart += 1
    for i in range(istart, len(args)):
        if not isinstance(args[i], str):
            raise TypeError("Expected string parameters")
        if args[i] == 'OutputView':
            if i >= len(args):
                raise ValueError("no value provided for OutputView")
            val = args[i+1]
            if not isinstance(val, str):
                raise TypeError("value for OutputView must be string")
            if val == 'full':
                mode = 0
            else:
                if val == 'same':
                    mode = 1.0
                else:
                    raise ValueError("invalid value for OutputView")
    return res, mode


def invfun(ss, theta, radius):
    m = np.tan(theta)
    r = np.zeros_like(m)
    ss = np.array(ss)
    poly_coef = np.flip(ss)
    poly_coef_tmp = poly_coef
    for j in range(len(m)):
        poly_coef_tmp[-2] = poly_coef[-2] - m[j]
        rhotmp = np.roots(poly_coef_tmp)
        res = rhotmp[np.where(
            np.imag(rhotmp) == 0 and rhotmp > 0 and rhotmp < radius)]
        if len(res) == 0 or len(res) > 1:
            r[j] = np.inf
        else:
            r[j] = res
    return r


def findinvpoly2(ss, radius, n):
    theta = range(-np.pi/2, 1.20, 0.01)
    r = invfun(ss, theta, radius)
    ind = np.where(r != np.inf)
    r = r[ind]

    pol = np.polyfit(theta, r, n)
    err = np.abs(r - np.polyval(pol, theta))
    return pol, err, n


def findinvpoly(ss, radius, *args):
    if len(args) == 0:
        maxerr = np.inf
        n = 1
        while maxerr > 0.01:
            # repeat until the reprojection error is smaller than 0.01 pixels
            n += 1
            pol, err = findinvpoly2(ss, radius, n)
            maxerr = max(err)
        pol, err, n = findinvpoly2(ss, radius, n)
    return pol, err, n


def getpoint(ss, m):
    """Given an image point it returns the 3D coordinates of its correspondent optical
    ray
    """
    w = np.array([-m[0, :], -m[1, :], np.polyval(np.flip(ss),
                                                 math.sqrt(np.power(m[0, :], 2) + np.power(m[1, :], 2)))]).reshape((3, 1))
    return w


def getpoint_(ss, m):
    """Given an image point it returns the 3D coordinates of its correspondent optical
    ray
    """
    w = np.array([m[0, :], m[1, :], np.polyval(np.flip(ss), math.sqrt(
        np.power(m[0, :], 2) + np.power(m[1, :], 2)))]).reshape((3, 1))
    return w


def cam2world(m, ocam_model):
    n_points = m.shape[1]

    ss = ocam_model['ss']
    xc = ocam_model['xc']
    yc = ocam_model['yc']
    c = ocam_model['c']
    d = ocam_model['d']
    e = ocam_model['e']

    aa = np.array([1, e, d, c]).reshape((2, 2))
    tt = np.array([yc, xc]).reshape((2, 1)) @ np.ones((1, n_points))
    m = aa**(-1) @ (m-tt)
    mm = getpoint_(ss, m)
    # normalizes coordinates so that they have unit length (projection onto the unit sphere)
    mm = normc(mm)
    return mm


def make_k(xmin, xmax, ymin, ymax, umin, umax, vmin, vmax):
    zmin, zmax = 1, 1
    fx = (umax*zmax - umin*zmin) / (xmax - xmin)
    fy = (vmax*zmax - vmin*zmin) / (ymax - ymin)
    cx = (zmin*umin - fx*xmin) / zmin
    cy = (zmin*vmin - fy*ymin) / zmin
    k = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]])
    return k


def make_normal_k(ocam_model, target_res):
    """make k such that top left and bottom right corners
       of distorted and undistorted image line up
    """
    scal = 20
    __image_corners = [np.array((1, 1)).reshape((2, 1)) + np.array((target_res[0]/scal, target_res[1]/scal)).reshape((2, 1)), np.array(
        (ocam_model.width, ocam_model.height)).reshape(2, 1) - np.array((target_res[0]/scal, target_res[1]/scal)).reshape((2, 1))]
    image_corners = np.array(__image_corners)
    xworld = cam2world(image_corners, ocam_model)
    x = [xworld[0, :] / xworld[2, :], xworld[1, :] /
         xworld[2, :], np.ones_like(xworld.shape[1])]
    x = np.array(x).reshape((3, 1))
    xmin, xmax = x[0, 0], x[0, 1]
    ymin, ymax = x[1, 0], x[1, 1]
    umin, umax = 1, target_res[0]
    vmin, vmax = 1, target_res[1]
    k = make_k(xmin, xmax, ymin, ymax, umin, umax, vmin, vmax)
    return k


def make_high_k(ocam_model, target_res):
    w = ocam_model['width']
    h = ocam_model['height']

    image_corners = np.array([[1, w/2, w, w/2], [h/2, 1, h/2, h]])
    xworld = cam2world(image_corners, ocam_model)

    thr = 0
    # if True:
    rowpix = np.array(np.array(
        [i+1 for i in range(w)]).reshape((1, w)), h/2 * np.ones((1, w))).squeeze(axis=1)
    colpix = np.array(np.round(w/2*1)*np.ones((1, h)),
                      np.array([i+1 for i in range(h)]).reshape((1, h))).squeeze(axis=1)
    xworld_row = cam2world(rowpix, ocam_model)
    xworld_col = cam2world(colpix, ocam_model)
    xworld_row = xworld_row[:, xworld_row[2, :] < thr]
    xworld_col = xworld_col[:, xworld_col[2, :] < thr]
    xworld = xworld[:, xworld[2, :] < -0.3]
    x_row = np.array([[xworld_row[0, :] / xworld_row[2, :]],
                      [xworld_row[1, :] / xworld_row[2, :]],
                      np.ones((1, xworld_row.shape[1]))])
    x_col = np.array([[xworld_col[0, :] / xworld_col[2, :]],
                      [xworld_col[1, :] / xworld_col[2, :]],
                      np.ones((1, xworld_col.shape[1]))])

    xmin, xmax = min(x_row[0, :]), max(x_row[0, :])
    ymin, ymax = min(x_col[1, :]), max(x_col[1, :])
    umin, umax = 1, target_res[0]
    vmin, vmax = 1, target_res[1]
    k = make_k(xmin, xmax, ymin, ymax, umin, umax, vmin, vmax)
    return k


def make_intrinsic_matrix(ocam_model, target_res, mode):
    if mode == 0:
        k = make_normal_k(ocam_model, target_res)
    else:
        if mode == 1:
            k = make_high_k(ocam_model, target_res)
        else:
            raise ValueError("invalid mode!")
    return k


def world2cam_fast(mm, ocam_model):
    ss = ocam_model['ss']
    xc = ocam_model['xc']
    yc = ocam_model['yc']
    c = ocam_model['c']
    d = ocam_model['d']
    e = ocam_model['e']
    pol = ocam_model['pol']

    npoints = mm.shape[1]
    theta = np.zeros((1, npoints))

    norm = math.sqrt(mm[0, :]**2 + mm[1, :]**2)

    # there are the scene points which are along the z-axis
    ind0 = np.where(norm == 0)
    norm[ind0] = 1e-6   # this will avoid division by ZERO later

    theta = math.atan(mm[2, :] / norm)
    # distance in pixel of the reprojected points from the image center
    rho = -np.polyval(pol, theta)
    x = mm[0, :] / norm * rho
    y = mm[1, :] / norm * rho
    # add center cordinates
    m = np.zeros_like(x)
    m[0, :] = x*1 + y*e + yc
    m[1, :] = x*d + y*c + xc
    return m


def ocam_undistort_map(ocam_model, *args):
    """ U = ocam_undistort_map(OCAM_MODEL[, UNDIST_RES, Name, Value, ...] )

     builds a undistortion map and calculates the perspective camera
     matrix K for a camera modeled with the OCamCalib toolbox

     --- input -----

     OCAM_MODEL  the calibration camera model as obtained from OCamCalib
                 toolbox
     UNDIST_RES  (optional) the resolution of the undistorted image
                 defaults to standard resolution
     Name, Value (optional) additional parametes
                 'OutputView'    'same' undistorted image is the same size
                                        as  distorted image
                                 'full' undistorted image contains all
                                        pixels of distorted image

     --- output -----

     U         structure with the following fields:

               U['map'] the undistortion map to be used with ocam_undistort
               U['K']   the intrinsic camera matrix K for a perspective camera,
                     same convention as matlab IntrinsicMatrix
               U['res'] target resolution
    """
    target_res, mode = get_res_and_mode(ocam_model, *args)
    # precompute inverse polynomail if necessary
    if not hasattr(ocam_model, 'pol'):
        width = ocam_model['width']
        height = ocam_model['height']
        ocam_model['pol'] = findinvpoly(
            ocam_model.ss, math.sqrt((width/2)**2 + (height/3)**2))
    assert (hasattr(ocam_model, 'pol'))
    # create an intrinsic matrix K that fits the required resolution and mode
    k = make_intrinsic_matrix(ocam_model, target_res, mode)
    # set up a rectangular even grid in the undistorted image pixel
    # coordinates. This grid corresponds to destination pixels
    # of the undistorted, perspective image
    nx, ny = np.meshgrid([i+1 for i in range(target_res[0])],
                         [j+1 for j in range(target_res[1])])
    # now shift and invert to go from pixel coordinates to homogenuous world coordinates
    n = np.array([[nx[:].T - k[0, 2]],
                  [ny[:].T - k[1, 2]]])
    x = (k[0:1, 0:1]**(-1)) * n
    xh = np.array([[-x[0, :]],
                   [-x[1, :]],
                   [-np.ones((1, target_res[0]*target_res[1]))]])
    # project the world coordinates back to distorted pixel coordinates to
    # get a fast lookup table
    U = dict()
    U['map'] = world2cam_fast(normc(xh), ocam_model).T
    U['K'] = k.T
    U['res'] = target_res
    return U


def im2double(im):
    info = np.iinfo(im.dtype)   # Get the data type of the input image
    # Divide all values by the largest possible value in the datatype
    return im.astype(np.float) / info.max


def im2uint8(img, *args, **kwargs):
    if isinstance(img, np.uint8):
        u = img
    elif isinstance(img, bool):
        u = np.uint16(img)  # img should be array but not list
        u[img] = 255
    else:
        if isinstance(img, np.int16):
            raise AttributeError("Invalid indexed image!")
        elif isinstance(img, np.uint16):
            if max(img[:]) > 255:
                raise AttributeError("Too many colors for 8 bit storage")
            else:
                u = np.uint8(img)
        else:   # double or single
            if max(img[:]) >= 257:
                raise AttributeError("Too many colors for 8 bit storage")
            elif min(img[:]) < 1:
                raise AttributeError("Invalid indexed image!")
            else:
                u = np.uint8(img - 1)


def interp2(x, y, img, xi, yi):
    """based on MATLAB function `interp2`
    """
    @numba.jit(nopython=True)
    def _interpolation(x, y, m, n, mm, nn, zxi, zyi, alpha, beta, img, return_img):
        qsx = int(m/2)
        qsy = int(n/2)
        for i in range(mm):
            for j in range(nn):
                # upper left coordinates
                zsx, zsy = int(zxi[i, j] + qsx), int(zyi[i, j] + qsy)
                # lower left coordinates
                zxx, zxy = int(zxi[i, j] + qsx), int(zyi[i, j] + qsy + 1)
                # upper right coordinates
                ysx, ysy = int(zxi[i, j] + qsx + 1), int(zyi[i, j] + qsy)
                # lower right coordinates
                yxx, yxy = int(zxi[i, j] + qsx + 1), int(zyi[i, j] + qsy + 1)
                fu0v = img[zsy, zsx] + alpha[i, j] * \
                    (img[ysy, ysx] - img[zsy, zsx])
                fu0v1 = img[zxy, zxx] + alpha[i, j] * \
                    (img[yxy, yxx] - img[zxy, zxx])
                fu0v0 = fu0v + beta[i, j] * (fu0v1 - fu0v)
                return_img[i, j] = fu0v0
        return return_img

    m, n = img.shape    # size of original matrix
    mm, nn = xi.shape   # size of smaller matrix
    zxi = np.floor(xi)
    zyi = np.floor(yi)
    alpha = xi - zxi
    beta = yi - zyi
    return_img = np.zeros((mm, nn))
    return_img = _interpolation(
        x, y, m, n, mm, nn, zxi, zyi, alpha, beta, img, return_img)
    return return_img


def interp2_(x, y, img, xi, yi):
    """
    final version: very nice
    x, y: prig coordinate
    img: src
    xi, yi:dsr coordinate

    bilinear interpolation:
    -----------------------------
    | q11(x1, y1) | q12(x1, y2) |
                q(x,y) 
    | q21(x2, y1) | q22(x2, y2) |
    -----------------------------
    f(x, y) = 1/(x2-x1)(y2-y1)*[f(x1,y1)*(x2-x)(y2-y))+f(x1, y2)*(x2-x)(y-y1)
              + f(x2, y1)*(x-x1)(y2-y)+f(x2, y2)*(x-x1)(y-y1)]

    (x2-x1)(y2-y1)=1 theorectically
    """
    cd = len(x)
    img_itp = np.ones([cd, cd])
    zxi = np.floor(xi).to(np.int16)  # upper left x
    zxi[zxi < 0] = 0
    zxi[zxi > cd-1] = cd-1
    syi = np.floor(yi).to(np.int16)  # upper left y
    syi[syi < 0] = 0
    syi[syi > cd-1] = cd-1
    yxi = np.floor(xi).to(np.int16)+1  # upper right x
    yxi[yxi < 0] = 0
    yxi[yxi > cd-1] = cd-1
    xyi = np.floor(yi).to(np.int16)+1  # upper right y
    xyi[xyi < 0] = 0
    xyi[xyi > cd-1] = cd-1

    for i in range(cd-1):   # lower coordinates
        for j in range(cd-1):
            x2_x = yxi[j, i] - xi[j, i]
            x_x1 = xi[j, i] - zxi[j, i]
            y2_y = xyi[j, i] - yi[j, i]
            y_y1 = yi[j, i] - syi[j, i]

            # q11: img[j, i], q12: img[j, i+1], q21: img[j+1, i],
            # q22: img[j+1, i+1]
            img_itp[j, i] = (img[syi[j, i], zxi[j, i]] * y2_y * x2_x +
                             img[syi[j, i], yxi[j, i]] * y2_y * x_x1 +
                             img[xyi[j, i], zxi[j, i]] * y_y1 * x2_x +
                             img[xyi[j, i], yxi[j, i]] * y_y1 * x_x1)

    return img_itp


def ocam_undistort(orig_img, U):
    w = U['res'][0]
    h = U['res'][1]
    if isinstance(orig_img, np.uint8):
        imd = im2double(orig_img)
    else:
        imd = orig_img
    if len(imd.shape) == 3:
        if isinstance(orig_img, np.uint8):
            undist = im2uint8(np.concatenate((np.reshape(interp2(U['map'][:, 0], U['map'][:, 1], imd[:, :, 0], U['map'][:, 0], U['map'][:, 1]), [w, h]),
                                              np.reshape(interp2(
                                                  U['map'][:, 0], U['map'][:, 1], imd[:, :, 1], U['map'][:, 0], U['map'][:, 1]), [w, h]),
                                              np.reshape(interp2(U['map'][:, 0], U['map'][:, 1], imd[:, :, 2], U['map'][:, 0], U['map'][:, 1]), [w, h]))).reshape((3, w, h)))
        else:
            undist = np.concatenate((np.reshape(interp2(U['map'][:, 0], U['map'][:, 1], imd[:, :, 0], U['map'][:, 0], U['map'][:, 1]), [w, h]),
                                     np.reshape(interp2(
                                         U['map'][:, 0], U['map'][:, 1], imd[:, :, 1], U['map'][:, 0], U['map'][:, 1]), [w, h]),
                                     np.reshape(interp2(U['map'][:, 0], U['map'][:, 1], imd[:, :, 2], U['map'][:, 0], U['map'][:, 1]), [w, h]))).reshape((3, w, h))
    elif len(imd.shape) == 2:
        undist = np.reshape(
            interp2(U['map'][:, 0], U['map'][:, 1], imd, U['map'][:, 0], U['map'][:, 1]))
    return undist


def comp_distortion(x_dist, k2):
    """
    %       [x_comp] = comp_distortion(x_dist,k2);
    %       
    %       compensates the radial distortion of the camera
    %       on the image plane.
    %       
    %       x_dist : the image points got without considering the
    %                radial distortion.
    %       x : The image plane points after correction for the distortion
    %       
    %       x and x_dist are 2xN arrays
    %
    %      Note : This compensation has to be done after the substraction
    %              of the center of projection, and division by the focal
    %              length.
    %       
    %       (do it up to a second order approximation)
    """
    two, N = x_dist.shape
    if two != 2:
        raise ValueError("The dimension of the points should be 2xN")
    if len(k2) > 1:
        x_comp = comp_distortion_oulu(x_dist, k2)
    else:
        radius_2 = x_dist[0, :]**2 + x_dist[1, :]**2
        radial_distortion = 1 + np.ones((2, 1))*(k2 * radius_2)
        radius_2_comp = (x_dist[0, :]**2 + x_dist[1, :]
                         ** 2) / radial_distortion[0, :]
        radial_distortion = 1 + np.ones((2, 1))*(k2 * radius_2_comp)
        x_comp = x_dist / radial_distortion
    return x_comp


def comp_distortion_oulu(xd, k):
    """
    %comp_distortion_oulu
    %
    %[x] = comp_distortion_oulu(xd,k)
    %
    %Compensates for radial and tangential distortion. Model From Oulu university.
    %For more informatino about the distortion model, check the forward projection mapping function:
    %project_points.m
    %
    %INPUT: xd: distorted (normalized) point coordinates in the image plane (2xN matrix)
    %       k: Distortion coefficients (radial and tangential) (4x1 vector)
    %
    %OUTPUT: x: undistorted (normalized) point coordinates in the image plane (2xN matrix)
    %
    %Method: Iterative method for compensation.
    %
    %Note: This compensation has to be done after the subtraction
    %      of the principal point, and division by the focal length.
    """
    if len(k) == 1:
        x = comp_distortion(xd, k)
    else:
        k1 = k[0]
        k2 = k[1]
        k3 = k[4]
        p1 = k[2]
        p2 = k[3]

        x = xd  # initial guess

        for _ in range(20):
            r_2 = sum(x**2)
            k_radial = 1 + k1 * r_2 + k2 * r_2**2 + k3 * r_2**3
            delta_x = np.array([2*p1*x[0, :]*x[1, :] + p2*(r_2 + 2*x[0, :]**2),
                                p1 * (r_2 + 2*x[1, :]**2)+2*p2*x[0, :]*x[1, :]]).reshape((2, 1))
            x = (xd - delta_x) / (np.ones((2, 1)) * k_radial)
    return x


def normalize_pixel(x_kk, fc=np.array([1, 1]).reshape((2, 1)), cc=np.array([0, 0]).reshape((2, 1)), kc=np.array([0, 0, 0, 0, 0]).reshape((5, 1)), alpha_c=0):
    """
     %normalize
    %
    %[xn] = normalize_pixel(x_kk,fc,cc,kc,alpha_c)
    %
    %Computes the normalized coordinates xn given the pixel coordinates x_kk
    %and the intrinsic camera parameters fc, cc and kc.
    %
    %INPUT: x_kk: Feature locations on the images
    %       fc: Camera focal length
    %       cc: Principal point coordinates
    %       kc: Distortion coefficients
    %       alpha_c: Skew coefficient
    %
    %OUTPUT: xn: Normalized feature locations on the image plane (a 2XN matrix)
    %
    %Important functions called within that program:
    %
    %comp_distortion_oulu: undistort pixel coordinates.
    """
    # First: Subtract principal point, and divide by the focal length
    x_distort = np.array(
        [(x_kk[0, :] - cc[0])/fc[0], (x_kk[1, :] - cc[1])/fc[1]]).reshape((2, 1))
    # Second: undo skew
    x_distort[0, :] = x_distort[0, :] - alpha_c * x_distort[1, :]

    if np.linalg.norm(kc) != 0:
        # Third: Compensate for lens distortion:
        xn = comp_distortion_oulu(x_distort, kc)
    else:
        xn = x_distort
    return xn


def rigid_motion(X, om=np.zeros((3,1)), T=np.zeros((3,1))):
    R, dRdom = cv2.Rodrigues(om)
    m, n = X.shape
    Y = R*X + np.matlib.repmat(T, 1, n)




def project_points2(X, om=np.zeros((3,1)), T=np.zeros((3,1)), f=np.zeros((2,1)), c=np.zeros((2,1)), k=np.zeros((5,1)), alpha=0):
    m, n = X.shape
    Y, dYdom, dYdT = rigid_motion(X, om, T)


def get_rectify_plane(stereo_param, img_size, *args):
    if len(args) == 0:
        replaceF = []
    elif len(args) == 1:
        replaceF = args[0]
    else:
        raise ValueError("Too many input arguments")

    with open(stereo_param, 'r') as f:
        param = json.load(f)

    focLeft = param['focLeft']
    focRight = param['focRight']
    cenLeft = param['cenLeft']
    cenRight = param['cenRight']
    alphaLeft = param['alphaLeft']
    alphaRight = param['alphaRight']
    kcLeft = param['kcLeft']
    kcRight = param['kcRight']
    rotVecRef = param['rotVecRef']
    transVecRef = param['transVecRef']

    # bring the 2 cameras in the same orientation by rotating them "minimally"
    # rotation converting right camera frame to common frame
    rotMatCFR = cv2.Rodrigues(-rotVecRef/2)
    rotMatCFL = rotMatCFR.T  # rotation converting left camera frame to common frame

    # vector of epipolar line in common rectangular frame
    eppVec = rotMatCFR * transVecRef

    # Rotate both cameras so as to bring the translation vector in alignment
    if np.abs(eppVec[0]) > np.abs(eppVec[1]):
        uu = np.array([1, 0, 0]).reshape((3, 1))  # Horizontal epipolar lines
        stereoType = 0
    else:
        uu = np.array([0, 1, 0]).reshape((3, 1))
        stereoType = 1
    if eppVec @ uu < 0:
        uu = -uu
    ww = np.cross(eppVec, uu)
    ww = ww / np.linalg.norm(ww)    # rotation axis
    rotVecEpp = np.arccos(np.abs((eppVec @ uu) / (np.linalg.norm(eppVec)
                                                  * np.linalg.norm(uu)))) * ww   # acos is angle to rotate
    rotMatEpp = cv2.Rodrigues(rotVecEpp)
    # Global rotations to be applied to both views:
    rotMatLeft = rotMatEpp * rotMatCFL
    rotMatRight = rotMatEpp * rotMatCFR

    imgH = img_size[0]
    imgW = img_size[1]

    # Computation of the *new* intrinsic parameters for both left and right
    # cameras:
    # Vertical focal length *MUST* be the same for both images (here, we are
    # trying to find a focal length that retains as much information contained
    # in the original distorted images):
    if kcLeft[0] < 0:
        focYLeftNew = focLeft[1] * (1 + kcLeft[0]
                                    * (imgW**2 + imgH**2)/(4*focLeft[1]**2))
    else:
        focYLeftNew = focLeft[1]
    if kcRight[0] < 0:
        focYRightNew = focRight[1] * (1 + kcRight[0]
                                      * (imgW**2 + imgH**2)/(4*focRight[1]**2))
    else:
        focYRightNew = focRight[1]

    focYNew = min(focYLeftNew, focYRightNew)
    if len(replaceF) != 0:
        focYNew = replaceF

    # For simplicity, let's pick the same value for the horizontal focal length
    # as the vertical focal length (resulting into square pixels):
    focLeftNew = round(np.array([[focYNew], [focYNew]]))
    focRightNew = round(np.array([[focYNew], [focYNew]]))

    # Select the new principal points to maximize the visible area in the
    # rectified images
    ptUndistLeft = normalize_pixel(np.array([[0, imgW-1, imgW-1, 0],
                                             [0, 0, imgH-1, imgH-1]]), focLeft, cenLeft, kcLeft, 0)
    cenLeftNew = np.array([(imgW-1)/2, (imgH-1)/2]).reshape((2, 1)) - \
        np.mean(project_points2(
            np.array([ptUndistLeft, [1, 1, 1, 1]]),
            cv2.Rodrigues(rotMatRight),
            np.zeros((3, 1)),
            focRightNew,
            np.array([[0], [0]]),
            np.zeros((5, 1)),
            0
        ),
        2)


def get_rectify_param(stereo_param, img_size):
    return
