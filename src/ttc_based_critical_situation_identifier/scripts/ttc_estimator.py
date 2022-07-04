#import rospy
#import ros_numpy
#from vehicle_msgs.msg import ADMAData
#from sensor_msgs.msg import PointCloud2
import numpy as np


class TTCEstimator:
    def __init__(self):
        self.adma_msg = None
        self.boundary = 1# read csv
       # rospy.Subscriber("/vehicle_marie/adma_data", ADMAData, self.adma_callback)
        #rospy.Subscriber("/vehicle_marie/velodyne_points", PointCloud2, self.pcd_callback)

    def adma_callback(self, msg):
        """
        self.global_T_lidar for every 10 messages (if timestamp % 100 ms == 0)
        """
        self.adma_msg = msg

        """
        ADMA 100 Hz -->
        
        
        
        """


    def pcd_callback(self, msg):
        """
        this call back runs ar 10 Hz
        """
        self.pcd_msg = msg

        """
        1. convert to numpy array
        2. transform to global coordinates: using self.global_T_lidar (find the cooresponding matrix)
        3. roi_removal (either z or ransac)
        4. 
        """

        #publish final estimated cluster (jsk bounding box)
        #publish ttc
        pass
    def roi_removal(self, data:np.array): # data: Nx4 (x, y, z, ref)

        return data # (M x 4)


    def z_based_ground_removal(self, data):
        return data

    def ransac_based_ground_removal(self, data):
        return data

    def knn(self, data):
        # return clusters
        pass
    def object_detection(self, clusters):
        # return cluster  # (for our car)
        pass
    def ttc_estimator(self, cluster, adma_information):
        pass
        # ml inference
        # return ttc
