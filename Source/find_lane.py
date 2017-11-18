"""
This file contains the methods that allow us to find the lanes in an image,
whether we have some previous information on the location of the lane
or not.
It is a slightly modified version of the code provided in the Udacity lesson.
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt

# This function allows us to find the lanes on an image that has already been processed
# to identify the likely lane pixels (and has been transformed to Bird's eye view)
# It is used when there is no previous information about the likely position of the lanes.
# It updates the position of the lanes and can plot the result.

def from_scratch(img, bplot=0):
	# Parameters
	n_win = 9 # Number of sliding windows	
	margin = 70 # Set the width of the windows +/- margin	
	minpix = 40	# Set minimum number of pixels found to recenter window

	# Straight from the lesson, we use the distribution of the edge pixels to identify
	# likely positions of the lanes.
	# Instead of taking the bottom half, we decided to take the bottom third of the image.
	histogram = np.sum(img[img.shape[0]//3:,:], axis=0)
	# Create an output image to draw on and  visualize the result
	out_img = np.dstack((img, img, img))*255
	# We use the left half mode as the likely position of the left hand lane line,
	# and the right half mode as the likely
	thirdpoint1 = np.int(histogram.shape[0]*0.33)
	thirdpoint2 = np.int(histogram.shape[0]*0.66)
	leftx_base = np.argmax(histogram[:thirdpoint1])
	rightx_base = np.argmax(histogram[thirdpoint2:]) + thirdpoint2	
	h_win = np.int(img.shape[0]/n_win) # Height of windows
	# Identify the x and y positions of all nonzero pixels in the image
	nonzero = img.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	# Current positions to be updated for each window
	leftx_current = leftx_base
	rightx_current = rightx_base
	
	# Create empty lists to receive left and right lane pixel indices
	left_lane_inds = []
	right_lane_inds = []

	# Step through the windows one by one
	for window in range(n_win):
		# Identify window boundaries in x and y (and right and left)
		win_y_low = img.shape[0] - (window+1)*h_win
		win_y_high = img.shape[0] - window*h_win
		win_xleft_low = leftx_current - margin
		win_xleft_high = leftx_current + margin
		win_xright_low = rightx_current - margin
		win_xright_high = rightx_current + margin
		# Draw the windows on the visualization image
		cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
		(0,255,0), 3) 
		cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
		(0,255,0), 3) 
		# Identify the nonzero pixels in x and y within the window
		good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
		(nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
		good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
		(nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
		# Append these indices to the lists
		left_lane_inds.append(good_left_inds)
		right_lane_inds.append(good_right_inds)
		# If you found > minpix pixels, recenter next window on their mean position
		if len(good_left_inds) > minpix:
		    leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
		if len(good_right_inds) > minpix:        
		    rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

	# Concatenate the arrays of indices
	left_lane_inds = np.concatenate(left_lane_inds)
	right_lane_inds = np.concatenate(right_lane_inds)
	# Extract left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds] 
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds] 
	# Fit a second order polynomial to each
	left_fit = np.polyfit(lefty, leftx, 2)
	right_fit = np.polyfit(righty, rightx, 2)
	# On top of that, we want to obtain a COMBINED FIT for both lanes
	# so we translate the right hand lane 
	aux = rightx + left_fit[2] - right_fit[2]
	fully = np.concatenate((lefty,righty))
	fullx = np.concatenate((leftx, aux))
	# Notice that we fit with x as a function of y (which makes sense since we are closer
	# to the vertical, and could easily have two values of y for a given x)
	full_fit = np.polyfit(fully, fullx, 2)
	left_fit[0] = 0.5*(left_fit[0]+full_fit[0])
	left_fit[1] = 0.5*(left_fit[1]+full_fit[1])
	right_fit[0] = 0.5*(right_fit[0]+full_fit[0])
	right_fit[1] = 0.5*(right_fit[1]+full_fit[1])


	# This is only to 
	if bplot == 1:
	# Generate x and y values for plotting
		ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
		left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
		right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

		out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
		out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
		plt.imshow(out_img)
		plt.plot(left_fitx, ploty, color='yellow')
		plt.plot(right_fitx, ploty, color='yellow')
		plt.xlim(0, 1280)
		plt.ylim(720, 0)

	return (left_fit, right_fit)

# This function is to be used if we have the parameters for the lane in the previous image
# after a sanity check to verify that the likely lanes are not too far from the previous ones
def from_previous(test, left_fit, right_fit, bplot=0):

	# Assume you now have a new warped binary image 
	# from the next frame of video (also called "binary_warped")
	# It's now much easier to find line pixels!
	nonzero = test.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	margin = 80
	left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
	left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
	left_fit[1]*nonzeroy + left_fit[2] + margin))) 

	right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
	right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
	right_fit[1]*nonzeroy + right_fit[2] + margin)))  

	# Again, extract left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds] 
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds]
	# Fit a second order polynomial to each
	left_fit = np.polyfit(lefty, leftx, 2)
	right_fit = np.polyfit(righty, rightx, 2)

	aux = rightx + left_fit[2] - right_fit[2]
	fully = np.concatenate((lefty,righty))
	fullx = np.concatenate((leftx, aux))

	full_fit = np.polyfit(fully, fullx, 2)

	# Generate x and y values for plotting
	ploty = np.linspace(0, test.shape[0]-1, test.shape[0] )
	left_fitx = 0.5*(left_fit[0]+full_fit[0])*ploty**2 + 0.5*(left_fit[1]+full_fit[1])*ploty + left_fit[2]
	right_fitx = 0.5*(right_fit[0]+full_fit[0])*ploty**2 + 0.5*(right_fit[1]+full_fit[1])*ploty + right_fit[2]
	
	if bplot==1:
		# Create an image to draw on and an image to show the selection window
		out_img = np.dstack((test, test, test))*255
		window_img = np.zeros_like(out_img)
		# Color in left and right line pixels
		out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
		out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]	
		# Generate a polygon to illustrate the search window area
		# And recast the x and y points into usable format for cv2.fillPoly()
		left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
		left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
				                      ploty])))])
		left_line_pts = np.hstack((left_line_window1, left_line_window2))
		right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
		right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
				                      ploty])))])
		right_line_pts = np.hstack((right_line_window1, right_line_window2))

		# Draw the lane onto the warped blank image
		cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
		cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
		result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
		plt.imshow(result)
		plt.plot(left_fitx, ploty, color='yellow')
		plt.plot(right_fitx, ploty, color='yellow')
		plt.xlim(0, 1280)
		plt.ylim(720, 0)

	return (left_fit, right_fit)


