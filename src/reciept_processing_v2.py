# -*- coding: utf-8 -*-
"""
Created on Wed May 26 16:31:03 2021

@author: asus
"""

import io, color 
import dct, idct
import filters, morphology, measure 
import binary_fill_holes
import pyplot as plt
import numpy as np
import transform
import math
import cluster
import distance
import Image
import cv3 as cv

# Reading Image from filesystem
img = io.imread("C:\\Users\\Rajeev\\timg1.jpeg")

# Step 1: Image Preprocessing
######################################################################################

#Converting image to grayscale to enhance processing speed
gray = color.rgb2gray(img)

#normalizing the global illumination by converting image from spatial domain into freq domain
frequencies = dct(dct(gray, axis=0), axis=1)

# then implementing High-pass filter to remove the slow illumination gradients
# Purpose is to even out global illumination
frequencies[:2,:2] = 0

# Converting the frequency domain data back into spacial domain
gray = idct(idct(frequencies, axis=1), axis=0)

# Normalizing the pixel magnitude between [0 to 1] 
gray = (gray - gray.min()) / (gray.max() - gray.min()) # renormalize to range [0:1]

# Plotting the normalizedd treated gray scale image
plt.imshow(gray)
plt.show()
 
# Step 2: Receipt Detection
######################################################################################

# Simple global threshold followed by some binary morphology and blob detection
# blurring the image using gaussian filter and then thresholding intensity at 60%
mask = filters.gaussian(gray, 2) > 0.6

# Detailed at "https://scikit-image.org/docs/dev/api/skimage.morphology.html#skimage.morphology.binary_closing"
# basically we are closing up (dark) gaps between (bright) features.
mask = morphology.binary_closing(mask, selem=morphology.disk(2, bool))

# this fills in the gaps left within the receipt to segment it from the background
mask = binary_fill_holes(mask, structure=morphology.disk(3, bool))
mask = measure.label(mask)
mask = (mask == 1 + np.argmax([r.filled_area for r in measure.regionprops(mask)]))

# Step 3: Corner Detection
######################################################################################

# Compute the outlines of the receipt from the foreground mask
edges = mask ^ morphology.binary_erosion(mask, selem=morphology.disk(2, bool))

# Applying probabilistic Hough transform to get the start and end points of the line segments in the image
segments = np.array(transform.probabilistic_hough_line(edges))
angles = np.array([np.abs(math.atan2(a[1]-b[1], a[0]-b[0]) - np.pi/2) for a,b in segments])

# Sorted the segments into horizontal and vertical segments
verticalSegments = segments[angles < np.pi/8] 
horizontalSegments = segments[angles >= np.pi/6]


# Definitions for Line
def line(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    return A, B, -C

# Definitions for Line Intersections
def lineIntersection(L1, L2):
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x,y
    else:
        return False
    

# Computing the intersection of each pair of horizontal and vertical segments
intersections = [lineIntersection(line(vs[0],vs[1]), line(hs[0],hs[1])) for vs in verticalSegments for hs in horizontalSegments]

# Estimate the bandwidth to use with the mean-shift algorithm.
bw = cluster.estimate_bandwidth(intersections, 0.1)

# Identifying corners by Mean shift clustering to discover “blobs” in a smooth density of samples , provides Coordinates of cluster centers.
corners = cluster.MeanShift(bandwidth=bw).fit(intersections).cluster_centers_

# Converting float to int
corners_ = np.int0(corners)

# identifying the top 4 corners in order to form a rectangle
coordinates = corners_[:4]

# # plotting the corner points on the gray image
# for i in coordinates:
#     x, y = i.ravel()
#     print(x,y)
#     cv.circle(gray, (x, y), 3, 255, 3)

# #plotting the gray scale image
# plt.imshow(gray), plt.show()


# Step 4:- Cropping
######################################################################################

# Pairwise distances between observations in n-dimensional space.
d = distance.pdist(coordinates)

# Calculate max width
w = int(max(d[0], d[5])) # = max(dist(p1, p2), dist(p3, p4))

# Calculate max height
h = int(max(d[2], d[3])) # = max(dist(p1, p4), dist(p2, p3))

# Using Projective Transform to remove the distortion mostly due to perspective (maps a rectangle to a quadrilateral)
tr = transform.ProjectiveTransform()

# Estimating the corner edges
tr.estimate(np.array([[0,0], [w,0], [0,h], [w,h] ]), coordinates)

# Applying perspective transformation through warping, then applying rotate for 90 degrees
receipt = transform.rotate(transform.warp(img, tr, output_shape=(h, w), order=1, mode='reflect'),90,resize=True)

# Plot Image
plt.imshow(receipt), plt.show()
plt.savefig('C:\\Users\\Rajeev\\receipt_op.jpeg')

im = Image.fromarray(receipt)
im.save('C:\\Users\\Rajeev\\receipt_gen.jpeg')
cv.imwrite('C:\\Users\\Rajeev\\receipt_gen.jpeg', receipt) 

