import numpy as np
import pandas as pd

class Projection:
    def __init__(self, img_width:int=1024, img_height:int=1024, focal_length:float=3000, avg_pixel_dev:float=0.)->None:

        self.im_x = img_width
        self.im_y = img_height
        self.flen = focal_length
        self.dev = avg_pixel_dev
        
        self.real_quat = self.__random_quat()
        self.frame = self.randomize()

        return

    def randomize(self, num_stars:int=np.random.randint(3, 10))->pd.DataFrame:
        
        # generate random unit quaternion in form [e1, e2, e3, n]
        self.real_quat:np.ndarray = self.__random_quat()

        # instantiate dataframe with random star positions
        frame:pd.DataFrame = self.__plot_random_stars(num_stars)

        # generate some centroid deviation (default 0)
        frame['DEV_X'] = np.random.normal(0, scale=self.dev, size=num_stars)
        frame['DEV_Y'] = np.random.normal(0, scale=self.dev, size=num_stars)

        # convert star image position to camera vectors (no error)
        frame['CV_REAL'] = frame.apply(self.__px_to_cv, axis=1, args=(False,))

        # convert approximated star position to camera vector (includes centroiding error)
        frame['CV_EST'] = frame.apply(self.__px_to_cv, axis=1, args=(True,))
        
        # get real ECI (this is always known if the star is correctly ID'd)
        frame['ECI_REAL'] = frame['CV_REAL'].apply(self.__quat_mult)

        self.frame = frame
        return frame

    def quat_to_ra_dec(self, q:np.ndarray)->np.ndarray:
        q1 = q[0]
        q2 = q[1]
        q3 = q[2]
        q4 = q[3]

        ra = np.arctan2(q2*q3 - q1*q4, q1*q3 + q2*q4)
        dec = np.arcsin(-q1**2*np.sqrt(q2**2 + q3**2 + q4**2))
        roll = np.arctan2(q2*q3 + q1*q4, -q1*q3 + q2*q4)

        angs = [ra, dec, roll]
        angDecs = [np.rad2deg(ang) for ang in angs]

        return angDecs
    
    def calc_diff(self, Q_calc:np.ndarray)->float:

        realRA = self.quat_to_ra_dec(self.real_quat)
        calcRA = self.quat_to_ra_dec(Q_calc)
        diff = [(a-b)*3600 for (a,b) in zip(realRA, calcRA)]

        return np.linalg.norm(diff)

    def __plot_random_stars(self, num:int)->pd.DataFrame:
        
        # generate N random (X,Y) tuples on the focal array
        imx = np.random.uniform(0, self.im_x, num)
        imy = np.random.uniform(0, self.im_y, num)

        return pd.DataFrame({'IMAGE_X': imx, 'IMAGE_Y': imy})

    def __px_to_cv(self, row:pd.Series, devFlag:bool)->np.ndarray:
        x = row['IMAGE_X']
        y = row['IMAGE_Y']
        
        if devFlag:
            x += row['DEV_X']
            y += row['DEV_Y']

        cvx = x - self.im_x/2
        cvy = y - self.im_y/2
        v = np.array([cvx, cvy, self.flen])
        return v/np.linalg.norm(v)

    def __random_quat(self)->np.ndarray:
        q = np.random.uniform(0, 1, 4)
        return q/np.linalg.norm(q)

    def __quat_mult(self, x:np.ndarray)->np.ndarray:
        e = -1*self.real_quat[:3]
        n = self.real_quat[3]
        
        Ca = (2*n**2 - 1) * np.identity(3)
        Cb = 2*np.outer(e, e)
        Cc = -2*n*self.__skew(e)

        return (Ca + Cb + Cc)@x
    
    def __skew(self, n:np.ndarray)->np.ndarray:
        return np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
