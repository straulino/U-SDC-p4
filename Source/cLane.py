"""
We define the class cLane
This class allows us to store the most recent lane parameters
and it also contains some useful methods (curvature, distance to centre)
"""

import numpy as np
from collections import deque

m_per_pix_y = 30/720 # meters per pixel in y dimension
m_per_pix_x = 3.7/720 # meteres per pixel in x dimension

class cLane():

    def __init__(self, img_size):
        
        self.detected = False  
        # parameters of up to the last 5 fittings
        self.coefficients = deque(maxlen=5)        
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #Image size for generating lane pixels
        self.img_size = img_size
        #x values for detected lane pixels
        self.allx = None  
        #y values for detected lane pixels
        self.ally = None
        #count number of times we didn't identify lanes
        self.dropped_frames = 0
        
    def compute_rad_curv(self):  
        points = self.get_points()
        y = points[:, 1]
        x = points[:, 0]
        fit_cr = np.polyfit(y * m_per_pix_y, x * m_per_pix_x, 2)
        return int(((1 + (2 * fit_cr[0] * self.img_size[0] * m_per_pix_y + fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * fit_cr[0]))

    def compute_dist_camera(self):
        fit = self.averaged_fit()
        lane = fit[2]+fit[1]*self.img_size[0]+fit[0]*(self.img_size[0]**2)   
        return np.absolute((self.img_size[1] // 2 - lane)) * m_per_pix_x
    
    def averaged_fit(self):    
        return np.array(self.coefficients).mean(axis=0)

    def get_points(self):
        y = np.linspace(0, self.img_size[0], self.img_size[1])
        fit = self.averaged_fit()
        return np.stack((fit[0] * y ** 2 + fit[1] * y + fit[2],y)).astype(np.int).T
