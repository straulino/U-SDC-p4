"""
This file contains all the routines we use to obtain the lanes in an image.
That includes undistorting the image, trying different colour channels,
using gradient as well thresholding for edge detection,
and applying the bird's eye transformation.
"""
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import os


# Undistort using global mtx and dist
def undistort(img, mtx, dist):
    return cv2.undistort(img, mtx, dist, None, mtx)

# Obtaining Bird's eye view (when using the right src and dst points)
def warp_image(img, src, dst):    
    len_Y, len_X = img.shape[:2]
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (len_X, len_Y))    
    return warped

# Draw lines to inspect accuracy of Bird's eye view
def draw_lines(img, points):
    cv2.line(img, (points[0][0], points[0][1]), (points[3][0], points[3][1]), color=[0, 0, 255], thickness=2)           
    cv2.line(img, (points[1][0], points[1][1]), (points[2][0], points[2][1]), color=[0, 0, 255], thickness=2)
    return img



# Cropping away none lane areas of the image
def crop_lanes(img, vertices):    
    #defining a blank mask to start with
    mask = np.zeros_like(img)       
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

# Get back an hsv based image that picks white and yellow
def hsvscale(img):    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sensitivity = 40
    yellow = cv2.inRange(hsv, (20, 70, 70), (30, 255, 255))
    white = cv2.inRange(hsv, (0, 0, 255-sensitivity), (255, sensitivity, 255))
    return cv2.bitwise_or(yellow, white)

# Returns a binary based on the direction of the gradient
# Image must have only one channel
def dir_threshold(img,  sobel_kernel=3, thresh=(0, np.pi/2)):    
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)   
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1    
    
    return binary_output

# Returns a binary based on the magnitude of the gradient
# Image must have only one channel
def mag_thresh(img, sobel_kernel=3, thresh=(0, 255)): 
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)  
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= thresh[0]) & (gradmag <= thresh[1])] = 1
    return binary_output

# Returns binary based on gradient on either axis
# Image must have only one channel
def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)): 
    if orient == 'y':
        sobel = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    else:
        sobel = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)    
    abssobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255*abssobel/np.max(abssobel))
    sbinary = np.zeros_like(scaled_sobel)
    sbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1    
    return sbinary

# For one channel images
def colour_thresh(img, thresh = (0, 255)):    
    binary = np.zeros_like(img)
    binary[(img > thresh[0]) & (img <= thresh[1])] = 1
    return binary


# This is the full pipeline for any given image
def obtain_edges(img, mtx, dist, vertices, ksize):
    # We begin by undistorting the image 
    img = undistort(img, mtx, dist)    
    
    # We will use both a greyscale version and the S channel of HSV      
    gray =  (0.5*img[:,:,0] + 0.4*img[:,:,1] + 0.1*img[:,:,2]).astype(np.uint8) 
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    s_channel = hls[:,:,2]
    #h_channel = hls[:,:,0]
    l_channel = hls[:,:,1]
    
    # We have used the Laplacian instead of the Sobel operator    
    laplacian_s = cv2.Laplacian(s_channel, cv2.CV_32F, ksize=21)
    mask_s = (laplacian_s < 0.15*np.min(laplacian_s)).astype(np.uint8)
    
    # Threshold on the s channel    
    #_, h_binary = cv2.threshold(h_channel.astype('uint8'), 150, 255, cv2.THRESH_BINARY)
    _, s_binary = cv2.threshold(s_channel.astype('uint8'), 150, 255, cv2.THRESH_BINARY)
    _, l_binary = cv2.threshold(l_channel.astype('uint8'), 70, 255, cv2.THRESH_BINARY)
    
    #Sobel on the gray image
    #_, gray_binary = cv2.threshold(gray.astype('uint8'), 150, 255, cv2.THRESH_BINARY)
    g_gradx = abs_sobel_thresh(gray, orient='x', sobel_kernel=ksize, thresh=(20, 255))
    #s_gradx = abs_sobel_thresh(s_channel, orient='x', sobel_kernel=ksize, thresh=(20, 255))
    
    #s_grady = abs_sobel_thresh(s_channel, orient='y', sobel_kernel=ksize, thresh=(30, 255))
    #s_mag_binary = mag_thresh(s_channel, sobel_kernel=ksize, thresh=(30, 255))
    #s_dir_binary = dir_threshold(s_channel, sobel_kernel=ksize, thresh=(0.8, 1.2))
    
    #s_combined = np.zeros_like(s_dir_binary)
    #s_combined[(s_gradx == 1) | ((s_mag_binary == 1) & (s_dir_binary == 1))] = 1
    
    #s_thresh = colour_thresh(s_channel, thresh = (120, 255))    
    hsv = hsvscale(img)
    #hsv1 = abs_sobel_thresh(hsv, orient='x', sobel_kernel=ksize, thresh=(30, 255))
    
    
    
    combined0 = np.clip(cv2.bitwise_or(s_binary, hsv), 0, 1).astype('uint8')
    combined1= np.clip(cv2.bitwise_or(g_gradx, mask_s), 0, 1).astype('uint8') 
    combined = np.clip(cv2.bitwise_or(combined0, combined1), 0, 1).astype('uint8')
    
    final = np.clip(cv2.bitwise_and(combined, l_binary), 0, 1).astype('uint8')
    
    return final



