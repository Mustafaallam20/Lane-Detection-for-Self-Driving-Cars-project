import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

def  abs_sobel_thresh(img_channel, orient='x', thresh=(20, 100), sobel_kernel=3):
    """
    Find edges that are aligned vertically and horizontally on the image

    :param img_channel: Channel from an image
    :param orient: Across which axis of the image are we detecting edges?
    :sobel_kernel: No. of rows and columns of the kernel (i.e. 3x3 small matrix)
    :return: Image with Sobel edge detection applied
    """
    # cv2.Sobel(input image, data type, prder of the derivative x, order of the
    # derivative y, small matrix used to calculate the derivative)
    if orient == 'x':
    # Will detect differences in pixel intensities going from 
        # left to right on the image (i.e. edges that are vertically aligned)
        sobel =  np.absolute(cv2.Sobel(img_channel, cv2.CV_64F, 1, 0, sobel_kernel))
    if orient == 'y':
        # Will detect differences in pixel intensities going from 
        # top to bottom on the image (i.e. edges that are horizontally aligned)
        sobel =  np.absolute(cv2.Sobel(img_channel, cv2.CV_64F, 0, 1, sobel_kernel))

    # Scale to 8-bit (0 - 255) then convert to type = np.uint8    
    scaled_sobel = np.uint8(255*sobel/np.max(sobel))
   
    # Create a binary mask where mag thresholds are met  
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 255

    # Return the result
    return binary_output

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    """
    #---------------------
    # This function takes in an image and optional Sobel kernel size, 
    # as well as thresholds for gradient magnitude. And computes the gradient magnitude, 
    # applies a threshold, and creates a binary output image showing where thresholds were met.
    #
    """
    # Take the gradient in x and y separately
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    
    # Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)

    # Create a binary mask where mag thresholds are met    
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 255

    # Return the binary image
    return binary_output

def dir_thresh(img, sobel_kernel=3, thresh=(0.7, 1.3)):
    """
    #---------------------
    # This function applies Sobel x and y, 
    # then computes the direction of the gradient,
    # and then applies a threshold.
    #
    """
    # Take the gradient in x and y separately
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    
    # Take the absolute value of the x and y gradients 
    # and calculate the direction of the gradient
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
   
    # Create a binary mask where direction thresholds are met 
    binary_output = np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 255
    
    # Return the binary image
    return binary_output.astype(np.uint8)

def get_combined_gradients(img, thresh_x, thresh_y, thresh_mag, thresh_dir):
    """
    #---------------------
    # This function isolates lane line pixels, by focusing on pixels
    # that are likely to be part of lane lines.
    # I am using Red Channel, since it detects white pixels very well. 
    #
    """
    rows, cols = img.shape[:2]
    
    # save cropped image for documentation
    temp = np.copy(img)
    temp = temp[:, 0:cols, 2]
    cv2.imwrite("./output_images/02_cropped.png", temp)

    R_channel = img[:, 0:cols, 2]   # focusing only on regions where lane lines are likely present

    sobelx = abs_sobel_thresh(R_channel, 'x', thresh_x)
    sobely = abs_sobel_thresh(R_channel, 'y', thresh_y)
    mag_binary = mag_thresh(R_channel, 3, thresh_mag)
    dir_binary = dir_thresh(R_channel, 15, thresh_dir)
    
    # debug
    #cv2.imshow('sobelx', sobelx)

    # combine sobelx, sobely, magnitude & direction measurements
    gradient_combined = np.zeros_like(dir_binary).astype(np.uint8)
    gradient_combined[((sobelx > 1) & (mag_binary > 1) & (dir_binary > 1)) | ((sobelx > 1) & (sobely > 1))] = 255  # | (R > 1)] = 255

    return gradient_combined

def channel_thresh(channel, thresh=(80, 255)):
    """
    #---------------------
    # This function takes in a channel of an image and
    # returns thresholded binary image
    # 
    """
    binary = np.zeros_like(channel)
    binary[(channel > thresh[0]) & (channel <= thresh[1])] = 255
    return binary

def get_combined_hls(img, th_h, th_l, th_s):
    """
    #---------------------
    # This function takes in an image, converts it to HLS colorspace, 
    # extracts individual channels, applies thresholding on them
    #
    """

    # convert to hls color space
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

    rows, cols = img.shape[:2]
    
    # trying to use Red channel info to improve results
    #R = img[220:rows - 12, 0:cols, 2]
    #_, R = cv2.threshold(R, 180, 255, cv2.THRESH_BINARY)
    
    H = hls[:, 0:cols, 0]
    L = hls[:, 0:cols, 1]
    S = hls[:, 0:cols, 2]

    h_channel = channel_thresh(H, th_h)
    l_channel = channel_thresh(L, th_l)
    s_channel = channel_thresh(S, th_s)
    
    # debug
    #cv2.imshow('Thresholded S channel', s_channel)

    # Trying to use Red channel, it works even better than S channel sometimes, 
    # but in cases where there is shadow on road and road color is different, 
    # S channel works better. 
    hls_comb = np.zeros_like(s_channel).astype(np.uint8)
    hls_comb[((s_channel > 1) & (l_channel == 0)) | ((s_channel == 0) & (h_channel > 1) & (l_channel > 1))] = 255 
    # trying to use both S channel and R channel
    #hls_comb[((s_channel > 1) & (h_channel > 1)) | (R > 1)] = 255
   
    # return combined hls image 
    return hls_comb

def combine_grad_hls(grad, hls):
    """ 
    #---------------------
    # This function combines gradient and hls images into one.
    # For binary gradient image, if pixel is bright, set that pixel value in reulting image to 255
    # For binary hls image, if pixel is bright, set that pixel value in resulting image to 255 
    # Edit: Assign different values to distinguish them
    # 
    """
    result = np.zeros_like(hls).astype(np.uint8)
    result[(grad > 1)] = 100
    result[(hls > 1)] = 255

    return result
