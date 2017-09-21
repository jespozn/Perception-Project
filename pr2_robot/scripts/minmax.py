#!/usr/bin/env python

import pcl
import numpy as np

cloud = pcl.load_XYZRGB("pass_through_filtered.pcd")
#pcl.getMinMax3D(cloud, min, max)
#print(cloud[0])
cloud_array = np.array(cloud)
#print(type(cloud_array))
print('min: ', np.amin(cloud_array, axis=0))
print('max: ', np.amax(cloud_array, axis=0))