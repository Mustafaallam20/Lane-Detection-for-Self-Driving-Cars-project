def draw_lane_lines(image):
    
    xsize, ysize = image.shape[1], image.shape[0]
    
    th_sobelx, th_sobely, th_mag, th_dir = (35, 100), (30, 255), (30, 255), (0.7, 1.3)
    th_h, th_l, th_s = (10, 100), (0, 60), (85, 255)
    
    combined_gradient = get_combined_gradients(image, th_sobelx, th_sobely, th_mag, th_dir)

    combined_hls = get_combined_hls(image, th_h, th_l, th_s)

    combined_result = combine_grad_hls(combined_gradient, combined_hls)

    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(combined_result, low_threshold, high_threshold)
    
    vertices = np.array([[(xsize//2-90,ysize//2+120),(100,ysize),(xsize-60, ysize),(xsize//2+130,ysize//2+120)]], dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)
    
    rho = 1 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 10     # minimum number of votes (intersections in Hough grid cell)
    min_line_len = 30 #minimum number of pixels making up a line
    max_line_gap = 20   # maximum gap in pixels between connectable line segments
    line_img = hough_lines(masked_edges, rho, theta, threshold, min_line_len, max_line_gap)
    
    img_with_lines = weighted_img(line_img, image, α=0.8, β=1., γ=0.)
    
    return img_with_lines
