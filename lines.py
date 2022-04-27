import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

def region_of_interest(img, vertices):

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

def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
    # Get the size of the figure
    xsize, ysize = img.shape[1], img.shape[0]
    
    # Fit two linear function for left/right lanes
    x_left = []
    y_left = []
    x_right = []
    y_right = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            if abs(x1-x2) == 0 or abs((y1-y2)/(x1-x2)) < 0.5:
                # Skip vertical line and the dot 
                break
                
            elif (y2-y1)/(x2-x1) > 0:
                # Right lane
                x_right.append(x2)
                x_right.append(x1)
                y_right.append(y2)
                y_right.append(y1)
            else:
                # Left lane
                x_left.append(x2)
                x_left.append(x1)
                y_left.append(y2)
                y_left.append(y1)
                
    y_start = ysize//2 + 120
    y_end = ysize
    if x_left:
        k_left, b_left = np.polyfit(x_left, y_left, 1)
        x_start_left = int((y_start-b_left)/k_left)
        x_end_left = int((y_end-b_left)/k_left)
        cv2.line(img, (x_start_left, y_start), (x_end_left, ysize), color, thickness)
    if x_right:
        k_right, b_right = np.polyfit(x_right, y_right, 1)
        x_start_right = int((y_start-b_right)/k_right)
        x_end_right = int((y_end-b_right)/k_right)
        cv2.line(img, (x_start_right, y_start), (x_end_right, ysize), color, thickness)
    
    
    try:
        center_lane = (x_end_right + x_end_left) / 2
        lane_width = x_end_right - x_end_left


        center_car = xsize / 2
        if center_lane > center_car:
            deviation = 'Vehicle is '+ str(round(abs(center_lane - center_car)*3.7/lane_width, 3)) + 'm Left of center'
        elif center_lane < center_car:
            deviation = 'Vehicle is '+ str(round(abs(center_lane - center_car)*3.7/lane_width, 3)) + 'm Right of center'
        else:
            deviation = 'by 0 (Centered)'

        cv2.putText(img, deviation, (10, 63), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (100, 100, 100), 1)
    except:pass
    
    try:
        contours = np.array([[x_end_left,ysize], [x_start_left,y_start], [x_start_right,y_start], [x_end_right,ysize]])
        cv2.fillPoly(img, pts = [contours], color =(0,200,0))
    except:pass

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

def weighted_img(img, initial_img, α=0.5, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)

