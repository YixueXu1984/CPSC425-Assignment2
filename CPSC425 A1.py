# import the packages we need for this assignment
from PIL import Image
import numpy as np
import math
from scipy import signal
import time
import cv2

im = Image.open('dog.jpg')


im_cat = Image.open('0a_cat.bmp')


# PART 2

# Q1. Boxfilter
# first check if n is odd. If n is even, return a AssertError
# otherwise return a n x n 2D-array of value 1/n^2
def boxfilter(n):
    assert n % 2 != 0, "Dimension must be odd."
    return np.full((n, n), float(1 / (n * n)))


# Q2. 1D Gaussian filter
def gauss1d(sigma):
    # array length: 6*sigma round up to next odd number
    length = (math.ceil(6 * sigma)) // 2
    x_axis = np.arange(-length, length + 1)
    # calculate Gaussian filter by the Gaussian function, exp(- x^2 / (2*sigma^2))
    result = np.exp(- np.square(x_axis) / (2 * np.square(sigma)))
    # normalize the value
    return np.array((result) / (np.sum(result)))


# Q3. 2D Gaussian filter
def gauss2d(sigma):
    # convert the 1D array to 2D
    gauss2d_origin = gauss1d(sigma)[np.newaxis]
    # the transpose of 1D Gaussian filter
    gauss2d_trans = gauss2d_origin.T
    # get the 2D Gaussian filter by the function 'convolve2d'
    result = signal.convolve2d(gauss2d_origin, gauss2d_trans)
    return result


# Q4. a) convolve2d_manual(array, filter)
def convolve2d_manual(array, filter):

    X = array
    K = filter

    # create a buffer 2d-array with zero-padding around original image
    x_offset = len(filter)//2
    y_offset = len(filter)//2
    c_buffer = np.zeros((len(array) + ((len(filter)//2)*2),len(list(zip(*array))) + ((len(filter)//2)*2)))
    c_buffer[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset] = X
    # print(c_buffer)

    # create the manually coded filtered image, fill with zero
    # same size as array
    c_manual = np.zeros_like(X)

    # filling the c_manual 2d-array with convolved filter computation results
    # example: top-left coner with ksize = 3
    # c_manual[0,0] = (c_buffer[0:3, 0:3] * K[::-1,::-1]).sum()
    for x in range(len(c_manual)):
        for y in range(len(list(zip(*c_manual)))):
            c_manual[x,y] = (c_buffer[x:x+len(filter), y:y+len(filter)] *K[::-1,::-1]).sum()
    # print(c_manual)

    return c_manual

# Q4. b)
def gaussconvolve2d_manual(array, sigma):
    filter = gauss2d(sigma)
    result = convolve2d_manual(array,filter)

    return result

# Q5. a)
def gaussconvolve2d_scipy(array, sigma):
    filter = gauss2d(sigma)
    result = signal.convolve2d(array, filter, 'same')
    return result


# Q4. c) & d), Q5. c), Q6.
def grayGaussImage():
    # print(im.size, im.mode, im.format)
    im_gray = im.convert('L')
    im_array = np.asarray(im_gray)
    # enable the corresponding Gaussian filter (manual / scipy)
    t1 = time.time()
    # result = gaussconvolve2d_manual(im_array, 10)
    result = gaussconvolve2d_scipy(im_array,10)
    duration = time.time() - t1
    # print(duration)
    image = Image.fromarray(result)
    return image


# im.show()
# grayGaussImage().show()


# Part 3

# Q1

def blurImage(img):
    im1 = Image.open(img)
    im1.show()

    # blurred Gaussian image

    im_array = np.asarray(im1)

    # separate RGB channel and store in different arrays
    im_red = im_array[:, :, 0]
    im_green = im_array[:, :, 1]
    im_blue = im_array[:, :, 2]

    # blur on every RGB channel
    im_rblur = gaussconvolve2d_scipy(im_red, 3)
    im_gblur = gaussconvolve2d_scipy(im_green, 3)
    im_bblur = gaussconvolve2d_scipy(im_blue, 3)

    # stack back and convert into image
    im_stack = np.dstack((im_rblur, im_gblur, im_bblur))
    im_stack = im_stack.astype('uint8')
    im_result = Image.fromarray(im_stack)

    # show image on screen
    # im_result.show()
    return im_result

# Q2. high freq image
def highfreq(img):

    im1 = Image.open(img)
    im1.show()

    # blurred Gaussian image

    im_array = np.asarray(im1)

    # separate RGB channel and store in different arrays
    im_red = im_array[:, :, 0]
    im_green = im_array[:, :, 1]
    im_blue = im_array[:, :, 2]

    # blur on every RGB channel
    im_rblur = gaussconvolve2d_scipy(im_red, 3)
    im_gblur = gaussconvolve2d_scipy(im_green, 3)
    im_bblur = gaussconvolve2d_scipy(im_blue, 3)

    # subtracting low freq Gaussian filtered image from the original
    im_hfred = np.subtract(im_red, im_rblur)
    im_hfgreen = np.subtract(im_green, im_gblur)
    im_hfblue = np.subtract(im_blue, im_bblur)

    # add 128
    im_hstack = np.dstack((im_hfred, im_hfgreen, im_hfblue))
    hstack = np.add(im_hstack, np.full(im_hstack.shape, 128))

    # add 0.5 for vis
    hstack = hstack.astype('uint8')
    for array in hstack:
        array = map(lambda x: x + 0.5, array)

    # show high frequency filtered image for Q2
    im_hresult = Image.fromarray(hstack)
    im_hresult.show()

    return im_hresult


# Q3. mixed image
def mixImage(img1, img2, sigma):
    im1 = Image.open(img1)
    im2 = Image.open(img2)

    im1_array = np.asarray(im1)
    im2_array = np.asarray(im2)

    # separate RGB channel and store in different arrays
    im1_red = im1_array[:, :, 0]
    im1_green = im1_array[:, :, 1]
    im1_blue = im1_array[:, :, 2]

    im2_red = im2_array[:, :, 0]
    im2_green = im2_array[:, :, 1]
    im2_blue = im2_array[:, :, 2]

    # blur on every RGB channel
    im1_rblur = gaussconvolve2d_scipy(im1_red, sigma)
    im1_gblur = gaussconvolve2d_scipy(im1_green, sigma)
    im1_bblur = gaussconvolve2d_scipy(im1_blue, sigma)

    im2_rblur = gaussconvolve2d_scipy(im2_red, sigma)
    im2_gblur = gaussconvolve2d_scipy(im2_green, sigma)
    im2_bblur = gaussconvolve2d_scipy(im2_blue, sigma)

    # subtracting low freq Gaussian filtered image from the original
    im2_hfred = np.subtract(im2_red, im2_rblur)
    im2_hfgreen = np.subtract(im2_green, im2_gblur)
    im2_hfblue = np.subtract(im2_blue, im2_bblur)

    # produce mixed frequency image for Q3
    im_mixred = np.add(im2_hfred, im1_rblur)
    im_mixgreen = np.add(im2_hfgreen, im1_gblur)
    im_mixblue = np.add(im2_hfblue, im1_bblur)

    im_mix = np.dstack((im_mixred, im_mixgreen, im_mixblue))
    im_mix = im_mix.astype('uint8')
    im_mresult = Image.fromarray(np.clip(im_mix, 0, 255))
    im_mresult.show()

    return im_mresult

# Part 4

def denoise():
    # read image
    # src = cv2.imread('box_gauss.png', 0)
    src = cv2.imread('box_speckle.png', 0)

    # apply guassian blur on src image
    # comment out corresponding unnecessary filter
    # leave all 3 for combined situation
    dst = cv2.GaussianBlur(src, ksize=(7, 7), sigmaX=50)

    dst = cv2.bilateralFilter(src, 7, sigmaColor=150, sigmaSpace=150)

    dst = cv2.medianBlur(src,7)

    cv2.imshow("combine", dst)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()

    return
