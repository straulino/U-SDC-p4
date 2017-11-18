import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import os

def calibrate()
	# termination criteria
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
	# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
	objp = np.zeros((6*9,3), np.float32)
	objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
	# Arrays to store object points and image points from all the images.
	objpoints = [] # 3d points in real world space
	imgpoints = [] # 2d points in image plane.
	# Make a list of calibration images
	calibration_dir = 'camera_cal'
	example_dir = 'examples'
	images = [os.path.join(calibration_dir, x) for x in os.listdir(calibration_dir)]
	# Step through the list and search for chessboard corners
	notfound = []
	notfoundname = []
	image_size = (cv2.imread(images[0]).shape[0], cv2.imread(images[0]).shape[1]) 
	for fname in images:
		img = cv2.imread(fname)
		img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)    
		img = cv2.equalizeHist(img)

		# Find the chessboard corners
		ret, corners = cv2.findChessboardCorners(img, (9,6),None)

		# If found, add object points, image points
		if ret == True:
		    objpoints.append(objp)
		    imgpoints.append(corners)

		    # Draw and save the corners
		    corners2 = cv2.cornerSubPix(img,corners,(11,11),(-1,-1),criteria)
		    img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
		    
		    cv2.imwrite(example_dir + "/drawn" +fname[22:],img)        
		else:
		    notfound.append(img)
		    notfoundname.append(fname)
	if len(notfound) > 0:
		print("Could not find chessboard in " + str(len(notfound)) + " of the images.")
		image_string = ""
		for x in notfoundname:
		    image_string += " " + x[11:] + ","
		print("These images are: " + image_string)
		
	else:
		print("Sucessfully found chessboards in all the images.")
	cv2.destroyAllWindows()

	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_size, None, None)
	return mtx, dist
