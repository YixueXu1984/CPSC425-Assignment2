from PIL import Image, ImageDraw
import numpy as np
import math
from scipy import signal
from scipy.ndimage import gaussian_filter

import ncc

scaleFactor = 0.75
family = Image.open("faces/family.jpg")
fans = Image.open('faces/fans.jpg')
judy = Image.open('faces/judybats.jpg')
sports = Image.open('faces/sports.jpg')
students = Image.open('faces/students.jpg')
tree = Image.open('faces/tree.jpg')
noel = Image.open('faces/599828192.png')
ttt = Image.open('faces/template.jpg')

orchard = Image.open('orchid.jpg')
violet = Image.open('violet.jpg')


# Part 1 Q2
def MakeGaussianPyramid(image, scale, minsize, color):
    x, y = image.size
    pyramid = [image]
    # stop when the size of image is smaller or equal to minsize
    while (x * scale > minsize) and (y * scale > minsize):
        x = int(x * scale)
        y = int(y * scale)

        im_array = np.asarray(image)

        # apply Gaussian filter
        if color == 'L':

            im_array = gaussian_filter(im_array, sigma=1 / (2 * scale))
            image = Image.fromarray(im_array)

        else:
            im_red = im_array[:, :, 0]
            im_green = im_array[:, :, 1]
            im_blue = im_array[:, :, 2]
            im_rblur = gaussian_filter(im_red, sigma=1 / (2 * scale))
            im_gblur = gaussian_filter(im_green, sigma=1 / (2 * scale))
            im_bblur = gaussian_filter(im_blue, sigma=1 / (2 * scale))
            im_stack = np.dstack((im_rblur, im_gblur, im_bblur))
            im_stack = im_stack.astype('uint8')
            image = Image.fromarray(im_stack)

        # make smaller image and add to list
        image = image.resize((x, y), Image.BICUBIC)
        pyramid.append(image)
    return pyramid


pFamily = MakeGaussianPyramid(family, 0.75, 15, 'L')
pFans = MakeGaussianPyramid(fans, 0.75, 15, 'L')
pJudy = MakeGaussianPyramid(judy, 0.75, 20, 'L')
pSports = MakeGaussianPyramid(sports, 0.75, 15, 'L')
pStudents = MakeGaussianPyramid(students, 0.75, 15, 'L')
pTree = MakeGaussianPyramid(tree, 0.75, 15, 'L')
pNoel = MakeGaussianPyramid(noel, 0.75, 15, 'RGB')

pOrchard = MakeGaussianPyramid(orchard, 0.75, 15, 'RGB')
pViolet = MakeGaussianPyramid(violet, 0.75, 15, 'RGB')


# Part 1 Q3
def ShowGaussianPyramid(pyramid):
    width, height = pyramid[0].size
    i = 0
    resultWidth = 0

    for image in pyramid:
        widthTmp, heightTmp = image.size
        resultWidth = resultWidth + widthTmp
        i = i + 1

    im = Image.new("RGB", (int(resultWidth), height), 'white')
    offset_x = 0
    offset_y = 0

    for image in pyramid:
        width, height = image.size
        im.paste(image, (offset_x, offset_y))
        offset_x = offset_x + width
    im.show()


im_template = Image.open('faces/template.jpg')
templateSize = 15


# Part 1 Q4
def FindTemplate(pyramid, template, threshold):
    # resize the template
    originWidth, originHeight = template.size
    newHeight = int(originHeight * 15 / originWidth)
    template = template.resize((templateSize, newHeight), Image.BICUBIC)

    im = pyramid[0].copy().convert('RGB')

    # compute NCC for every image in the pyramid
    ncc_array = []
    for image in pyramid:
        cor = ncc.normxcorr2D(image, template)
        ncc_array.append(cor)

    k = 0

    for correlation in ncc_array:

        # for each pixel, if result > threshold, draw the boundary of template at the location
        for y in range(len(correlation)):
            for x in range(len(correlation[0])):
                if correlation[y][x] > threshold:
                    draw = ImageDraw.Draw(im)

                    # draw.line(x1,y1,x2,y2), fill="red", width = 2
                    # draw red line from (x1, y1) to (x2, y2)
                    draw.line(((x - templateSize / 2) / (0.75 ** k), (y - newHeight / 2) / (0.75 ** k),
                               (x + templateSize / 2) / (0.75 ** k), (y - newHeight / 2) / (0.75 ** k)), fill="red",
                              width=2)
                    draw.line(((x - templateSize / 2) / (0.75 ** k), (y - newHeight / 2) / (0.75 ** k),
                               (x - templateSize / 2) / (0.75 ** k), (y + newHeight / 2) / (0.75 ** k)), fill="red",
                              width=2)
                    draw.line(((x - templateSize / 2) / (0.75 ** k), (y + newHeight / 2) / (0.75 ** k),
                               (x + templateSize / 2) / (0.75 ** k), (y + newHeight / 2) / (0.75 ** k)), fill="red",
                              width=2)
                    draw.line(((x + templateSize / 2) / (0.75 ** k), (y + newHeight / 2) / (0.75 ** k),
                               (x + templateSize / 2) / (0.75 ** k), (y - newHeight / 2) / (0.75 ** k)), fill="red",
                              width=2)
                    del draw
        k = k + 1

    im.show()
    return 0


# Part 2 Q2
# reused code from Assignment 1 for high freq images
def MakeLaplacianPyramid(image, scale, minsize):
    pyramid_g = MakeGaussianPyramid(image, scale, minsize, 'RGB')
    length = len(pyramid_g)
    # L0 = G0 - expand(G1)
    pyramid_l = []

    for i in range(0, length - 1):
        w0, h0 = pyramid_g[i].size
        li = pyramid_g[i + 1].resize((w0, h0), Image.BICUBIC)

        im0_array = np.asarray(pyramid_g[i])
        im0_red = im0_array[:, :, 0]
        im0_green = im0_array[:, :, 1]
        im0_blue = im0_array[:, :, 2]

        im1_array = np.asarray(li)
        im1_red = im1_array[:, :, 0]
        im1_green = im1_array[:, :, 1]
        im1_blue = im1_array[:, :, 2]

        im_hfred = np.subtract(im0_red, im1_red)
        im_hfgreen = np.subtract(im0_green, im1_green)
        im_hfblue = np.subtract(im0_blue, im1_blue)

        im_hstack = np.dstack((im_hfred, im_hfgreen, im_hfblue))
        im_hresult = Image.fromarray(im_hstack)
        pyramid_l.append(im_hresult)

    pyramid_l.append(pyramid_g[length - 1])
    return pyramid_l


# Part 2 Q3
# add 128 and normalize for visualization
def ShowLaplacianPyramid(pyramid):
    pyramid_result = []
    length = len(pyramid)

    for image in pyramid[:length - 1]:
        im_array = np.asarray(image)
        hstack = np.add(im_array, np.full(im_array.shape, 128))
        hstack = hstack.astype('uint8')
        for array in hstack:
            array = map(lambda x: x + 0.5, array)
        im_hresult = Image.fromarray(hstack)
        pyramid_result.append(im_hresult)

    pyramid_result.append(pyramid[len(pyramid) - 1])

    width, height = pyramid[0].size
    i = 0
    resultWidth = 0

    for image in pyramid:
        widthTmp, heightTmp = image.size
        resultWidth = resultWidth + widthTmp
        i = i + 1

    im = Image.new("RGB", (int(resultWidth), height), 'white')
    offset_x = 0
    offset_y = 0

    for image in pyramid_result:
        width, height = image.size
        im.paste(image, (offset_x, offset_y))
        offset_x = offset_x + width
    im.show()
    return

# FindTemplate(pJudy, im_template,0.7)


lNoel = MakeLaplacianPyramid(noel, 0.75, 15)
lOrchard = MakeLaplacianPyramid(orchard, 0.75, 100)
lViolet = MakeLaplacianPyramid(violet, 0.75, 100)

# ShowGaussianPyramid(pFamily)

ShowLaplacianPyramid(lOrchard)
ShowLaplacianPyramid(lViolet)


# ShowGaussianPyramid(pNoel)


# Part 2 Q4
# Recursively reconstruct Gaussian pyramid
def ReconstructGaussianFromLaplacianPyramid(lPyramid):
    # acquire the depth of recursive tree
    length = len(lPyramid)
    lPyramid_copy = lPyramid
    RecursivelyReconstructGaussianPyramid(length - 1, lPyramid_copy)
    return lPyramid_copy


def RecursivelyReconstructGaussianPyramid(depth, lPyramid):
    # base case: when reach the lowest level(lowest resolution)
    # return the same image of Gaussian pyramid
    if depth == 0:
        return lPyramid[len(lPyramid) - 1]

    # otherwise: recursively resize the deeper Laplacian image and add to current level image
    # to reconstruct Gaussian pyramid
    else:
        RecursivelyReconstructGaussianPyramid(depth - 1, lPyramid)
        length = len(lPyramid)
        im1 = lPyramid[length - 1 - depth]
        im0 = lPyramid[length - depth]
        x, y = im1.size
        im0 = im0.resize((x, y), Image.BICUBIC)
        im0_array = np.asarray(im0)
        im1_array = np.asarray(im1)

        # separate RGB channel and store in different arrays
        im0_red = im0_array[:, :, 0]
        im0_green = im0_array[:, :, 1]
        im0_blue = im0_array[:, :, 2]

        im1_red = im1_array[:, :, 0]
        im1_green = im1_array[:, :, 1]
        im1_blue = im1_array[:, :, 2]

        im_mixred = np.add(im0_red, im1_red)
        im_mixgreen = np.add(im0_green, im1_green)
        im_mixblue = np.add(im0_blue, im1_blue)

        im_mix = np.dstack((im_mixred, im_mixgreen, im_mixblue))
        im_mix = im_mix.astype('uint8')
        im_mresult = Image.fromarray(np.clip(im_mix, 0, 255))
        lPyramid[length - depth - 1] = im_mresult
        # print('updated lp[', length - depth - 1, ']')
        # lPyramid[length - depth - 1].show()

        return


# lNoel[6] = pNoel[6]
# llNoel = ReconstructGaussianFromLaplacianPyramid(lNoel)
# rpOrchard = ReconstructGaussianFromLaplacianPyramid(lOrchard)
# rpViolet = ReconstructGaussianFromLaplacianPyramid(lViolet)
# ShowGaussianPyramid(rpOrchard)
# ShowGaussianPyramid(rpViolet)

maskGreen = Image.open('orchid_mask.bmp')
pmaskGreen = MakeGaussianPyramid(maskGreen, 0.75, 100, 'L')


# ShowGaussianPyramid(pmaskGreen)

# Part 2 Q6
def composeImage(lPyramid1, lPyramid2, mask):
    lPyramid_result = []
    for i in range(0, len(lPyramid1)):
        image1 = lPyramid1[i]
        image2 = lPyramid2[i]
        mas = mask[i]
        image_result = Image.composite(image1, image2, mas)
        lPyramid_result.append(image_result)

    return lPyramid_result

# mixlGreen = composeImage(lOrchard,lViolet,pmaskGreen)
# ShowGaussianPyramid(mixlGreen)
# rpmixGreen = ReconstructGaussianFromLaplacianPyramid(mixlGreen)
# ShowGaussianPyramid(rpmixGreen)

# Part 2 Q7 - Cup

# blueCup = Image.open('blue_cup.jpg')
# greenCup = Image.open('green_cup.jpg')
# cupMask = Image.open('cup_mask.bmp')
#
# lBCup = MakeLaplacianPyramid(blueCup, 0.75, 50)
# lGCup = MakeLaplacianPyramid(greenCup, 0.75, 50)
# pCupMask = MakeGaussianPyramid(cupMask, 0.75, 50, 'L')
#
# mixCup = composeImage(lBCup, lGCup, pCupMask)
# rpmixCup = ReconstructGaussianFromLaplacianPyramid(mixCup)
# ShowGaussianPyramid(rpmixCup)

# Part 2 Q7 - tomato&apple

# tomato = Image.open('tomato.jpg')
# apple = Image.open('apple.jpg')
# tMask = Image.open('tomato_mask.bmp')
#
# lTomato = MakeLaplacianPyramid(tomato, 0.75, 25)
# lApple = MakeLaplacianPyramid(apple, 0.75, 25)
# pTMask = MakeGaussianPyramid(tMask, 0.75, 25, 'L')
#
# mixTomato = composeImage(lTomato, lApple, pTMask)
# rpmixTomato = ReconstructGaussianFromLaplacianPyramid(mixTomato)
# ShowGaussianPyramid(rpmixTomato)