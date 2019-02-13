import os
import cv2
import numpy as np
import math
def warpLocal(src, uv):
    '''
    Input:
        src --    source image in a numpy array with values in [0, 255].
                  The dimensions are (rows, cols, color bands BGR).
        uv --     warped image in terms of addresses of each pixel in the source
                  image in a numpy array.
                  The dimensions are (rows, cols, addresses of pixels [:,:,0]
                  are x (i.e., cols) and [:,,:,1] are y (i.e., rows)).
    Output:
        warped -- resampled image from the source image according to provided
                  addresses in a numpy array with values in [0, 255]. The
                  dimensions are (rows, cols, color bands BGR).
    '''
    width = src.shape[1]
    height  = src.shape[0]
    mask = cv2.inRange(uv[:,:,1],0,height-1)&cv2.inRange(uv[:,:,0],0,width-1)
    warped = cv2.remap(src, uv[:, :, 0].astype(np.float32),\
             uv[:, :, 1].astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    img2_fg = cv2.bitwise_and(warped,warped,mask = mask)
    return img2_fg


def computeSphericalWarpMappings(dstShape, f, k1, k2):
    '''
    Compute the spherical warp. Compute the addresses of each pixel of the
    output image in the source image.

    Input:
        dstShape -- shape of input / output image in a numpy array.
                    [number or rows, number of cols, number of bands]
        f --        focal length in pixel as int
                    See assignment description on how to find the focal length
        k1 --       horizontal distortion as a float
        k2 --       vertical distortion as a float
    Output:
        uvImg --    warped image in terms of addresses of each pixel in the
                    source image in a numpy array.
                    The dimensions are (rows, cols, addresses of pixels
                    [:,:,0] are x (i.e., cols) and [:,:,1] are y (i.e., rows)).
    '''

    # calculate minimum y value
    vec = np.zeros(3)
    vec[0] = np.sin(0.0) * np.cos(0.0)
    vec[1] = np.sin(0.0)
    vec[2] = np.cos(0.0) * np.cos(0.0)
    min_y = vec[1]

    # calculate spherical coordinates
    # (x,y) is the spherical image coordinates.
    # (xf,yf) is the spherical coordinates, e.g., xf is the angle theta
    # and yf is the angle phi
    one = np.ones((dstShape[0], dstShape[1]))
    xf = one * np.arange(dstShape[1])
    yf = one.T * np.arange(dstShape[0])
    yf = yf.T

    xf = ((xf - 0.5 * dstShape[1]) / f)
    yf = ((yf - 0.5 * dstShape[0]) / f - min_y)
    # BEGIN TODO 1
    # add code to apply the spherical correction, i.e.,
    # compute the Euclidean coordinates,
    # and project the point to the z=1 plane at (xt/zt,yt/zt,1),
    # then distort with radial distortion coefficients k1 and k2
    # Use xf, yf as input for your code block and compute xt, yt
    # as output for your code. They should all have the shape
    # (img_height, img_width)
    # TODO-BLOCK-BEGIN
    ## Compute the Euclidean coordinates

    x = np.sin(xf)*np.cos(yf)
    y = np.sin(yf)
    z = np.cos(xf)*np.cos(yf)

    ## Project
    x = x/z
    y = y/z
    z = 1

    ## Distort
    radius2 = x ** 2 + y ** 2
    xt = x * (1 + k1*radius2 + k2*radius2*radius2)
    yt = y * (1 + k1*radius2 + k2*radius2*radius2)
    # TODO-BLOCK-END
    # END TODO
    # Convert back to regular pixel coordinates
    xn = 0.5 * dstShape[1] + xt * f
    yn = 0.5 * dstShape[0] + yt * f
    uvImg = np.dstack((xn,yn))
    return uvImg


def warpSpherical(image, focalLength, k1=-0.21, k2=0.26):
    '''
    Input:
        image --       filename of input image as string
        focalLength -- focal length in pixel as int
                       see assignment description on how to find the focal
                       length
        k1, k2 --      Radial distortion parameters
    Output:
        dstImage --    output image in a numpy array with
                       values in [0, 255]. The dimensions are (rows, cols,
                       color bands BGR).
    '''

    # compute spherical warp
    # compute the addresses of each pixel of the output image in the
    # source image
    uv = computeSphericalWarpMappings(np.array(image.shape), focalLength, k1, \
        k2)

    # warp image based on backwards coordinates
    return warpLocal(image, uv)


