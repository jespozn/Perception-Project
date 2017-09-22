#!/usr/bin/env python

# Import modules
import numpy as np
import matplotlib.colors
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle
from sensor_stick.srv import GetNormals
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml


# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"]  = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict

# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)

# Voxel GRid Downsampling
def VGD(cloud):
    # Create a VoxelGrid filter object for our input point cloud
    vox = cloud.make_voxel_grid_filter()

    # Choose a voxel (also known as leaf) size
    LEAF_SIZE = 0.005

    # Set the voxel (or leaf) size
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)

    # Call the filter function to obtain the resultant downsampled point cloud
    cloud_filtered = vox.filter()

    return cloud_filtered

def stat_filter(cloud):
    # Create a statistical outlier filter object for our input point cloud
    outlier_filter = cloud.make_statistical_outlier_filter()

    # Set the number of neighboring points to analyze for any given point
    outlier_filter.set_mean_k(50)

    # Set threshold scale factor
    x = 0.4  # 1.0

    # Any point with a mean distance larger than global (mean distance+x*std_dev) will be considered outlier
    outlier_filter.set_std_dev_mul_thresh(x)

    # Finally call the filter function for magic
    cloud_filtered = outlier_filter.filter()

    return cloud_filtered

# PassThrough Filter
def passT_filter(cloud):
    # Create a PassThrough filter object.
    passthrough = cloud.make_passthrough_filter()

    # Assign axis and range to the passthrough filter object.
    filter_axis = 'z'
    passthrough.set_filter_field_name(filter_axis)
    axis_min = 0.5
    axis_max = 0.9
    passthrough.set_filter_limits(axis_min, axis_max)
    # Finally use the filter function to obtain the resultant point cloud.
    cloud_filtered = passthrough.filter()

    passthrough = cloud_filtered.make_passthrough_filter()
    filter_axis = 'y'
    passthrough.set_filter_field_name(filter_axis)
    axis_min = -0.46
    axis_max = 0.46
    passthrough.set_filter_limits(axis_min, axis_max)
    # Finally use the filter function to obtain the resultant point cloud.
    cloud_filtered = passthrough.filter()

    passthrough = cloud_filtered.make_passthrough_filter()
    filter_axis = 'x'
    passthrough.set_filter_field_name(filter_axis)
    axis_min = 0.34
    axis_max = 0.9
    passthrough.set_filter_limits(axis_min, axis_max)
    # Finally use the filter function to obtain the resultant point cloud.
    cloud_filtered = passthrough.filter()

    return cloud_filtered

# RANSAC plane segmentation
def RANSAC(cloud):
    # Create the segmentation object
    seg = cloud.make_segmenter()

    # Set the model you wish to fit
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)

    # Max distance for a point to be considered fitting the model
    max_distance = 0.01
    seg.set_distance_threshold(max_distance)

    # Call the segment function to obtain set of inlier indices and model coefficients
    inliers, coefficients = seg.segment()

    return inliers

def rgb_to_hsv(rgb_list):
    rgb_normalized = [1.0 * rgb_list[0] / 255, 1.0 * rgb_list[1] / 255, 1.0 * rgb_list[2] / 255]
    hsv_normalized = matplotlib.colors.rgb_to_hsv([[rgb_normalized]])[0][0]
    return hsv_normalized

def compute_color_histograms(cloud, using_hsv=False, nbins=32, bins_range=(0, 256)):
    # Compute histograms for the clusters
    point_colors_list = []

    # Step through each point in the point cloud
    for point in pc2.read_points(cloud, skip_nans=True):
        rgb_list = float_to_rgb(point[3])
        if using_hsv:
            point_colors_list.append(rgb_to_hsv(rgb_list) * 255)
        else:
            point_colors_list.append(rgb_list)

    # Populate lists with color values
    channel_1_vals = []
    channel_2_vals = []
    channel_3_vals = []

    for color in point_colors_list:
        channel_1_vals.append(color[0])
        channel_2_vals.append(color[1])
        channel_3_vals.append(color[2])

    # Compute histograms
    hist_channel_1 = np.histogram(channel_1_vals, bins=nbins, range=bins_range)
    hist_channel_2 = np.histogram(channel_2_vals, bins=nbins, range=bins_range)
    hist_channel_3 = np.histogram(channel_3_vals, bins=nbins, range=bins_range)

    # Concatenate and normalize the histograms
    hist_features = np.concatenate((hist_channel_1[0], hist_channel_2[0], hist_channel_3[0])).astype(np.float64)
    normed_features = hist_features / np.sum(hist_features)
    return normed_features

def compute_normal_histograms(normal_cloud, nbins=32, bins_range=(-1, 1)):
    norm_x_vals = []
    norm_y_vals = []
    norm_z_vals = []

    for norm_component in pc2.read_points(normal_cloud,
                                          field_names=('normal_x', 'normal_y', 'normal_z'),
                                          skip_nans=True):
        norm_x_vals.append(norm_component[0])
        norm_y_vals.append(norm_component[1])
        norm_z_vals.append(norm_component[2])

    # Compute histograms of normal values (just like with color)
    hist_channel_1 = np.histogram(norm_x_vals, bins=nbins, range=bins_range)
    hist_channel_2 = np.histogram(norm_y_vals, bins=nbins, range=bins_range)
    hist_channel_3 = np.histogram(norm_z_vals, bins=nbins, range=bins_range)

    # Concatenate and normalize the histograms
    hist_features = np.concatenate((hist_channel_1[0], hist_channel_2[0], hist_channel_3[0])).astype(np.float64)
    normed_features = hist_features / np.sum(hist_features)

    return normed_features

# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

    print 'pcl_msg received...'

# Exercise-2 TODOs:

    # Convert ROS msg to PCL data
    cloud = ros_to_pcl(pcl_msg)
    filename = 'original.pcd'
    pcl.save(cloud, filename)

    # Statistical Outlier Filtering
    outlier_filtered = stat_filter(cloud)
    filename = 'stat_filter.pcd'
    pcl.save(outlier_filtered, filename)

    # Voxel Grid Downsampling
    voxel_downsampled = VGD(outlier_filtered)
    filename = 'voxel_downsampled.pcd'
    pcl.save(voxel_downsampled, filename)

    # PassThrough Filter
    passT_filtered = passT_filter(voxel_downsampled)
    filename = 'pass.pcd'
    pcl.save(passT_filtered, filename)

    # RANSAC Plane Segmentation
    inliers = RANSAC(passT_filtered)

    # Extract inliers and outliers
    cloud_table = passT_filtered.extract(inliers, negative=False)
    filename = 'cloud_table.pcd'
    pcl.save(cloud_table, filename)
    cloud_objects = passT_filtered.extract(inliers, negative=True)
    filename = 'cloud_objects.pcd'
    pcl.save(cloud_objects, filename)

    # Euclidean Clustering
    white_cloud = XYZRGB_to_XYZ(cloud_objects)
    tree = white_cloud.make_kdtree()
    # Create a cluster extraction object
    ec = white_cloud.make_EuclideanClusterExtraction()
    # Set tolerances for distance threshold
    # as well as minimum and maximum cluster size (in points)
    ec.set_ClusterTolerance(0.01)
    ec.set_MinClusterSize(100)
    ec.set_MaxClusterSize(5000)
    # Search the k-d tree for clusters
    ec.set_SearchMethod(tree)
    # Extract indices for each of the discovered clusters
    cluster_indices = ec.Extract()

    # Create Cluster-Mask Point Cloud to visualize each cluster separately
    # Assign a color corresponding to each segmented object in scene
    cluster_color = get_color_list(len(cluster_indices))

    color_cluster_point_list = []

    for j, indices in enumerate(cluster_indices):
        for i, indice in enumerate(indices):
            color_cluster_point_list.append([white_cloud[indice][0],
                                             white_cloud[indice][1],
                                             white_cloud[indice][2],
                                             rgb_to_float(cluster_color[j])])

    # Create new cloud containing all clusters, each with unique color
    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)
    filename = 'cluster.pcd'
    pcl.save(cluster_cloud, filename)

    # Convert PCL data to ROS messages
    ros_cloud_objects = pcl_to_ros(cloud_objects)
    ros_cloud_table = pcl_to_ros(cloud_table)
    ros_cluster_cloud = pcl_to_ros(cluster_cloud)

    # Publish ROS messages
    pcl_objects_pub.publish(ros_cloud_objects)
    pcl_table_pub.publish(ros_cloud_table)
    pcl_cluster_cloud_pub.publish(ros_cluster_cloud)

# Exercise-3 TODOs:

    print('Cluster classification...')
    # Classify the clusters! (loop through each detected cluster one at a time)
    detected_objects_labels = []
    detected_objects = []

    for index, pts_list in enumerate(cluster_indices):
        # Grab the points for the cluster from the extracted outliers (cloud_objects)
        pcl_cluster = cloud_objects.extract(pts_list)
        # convert the cluster from pcl to ROS using helper function
        ros_cluster = pcl_to_ros(pcl_cluster)

        # Compute the associated feature vector
        chists = compute_color_histograms(ros_cluster, using_hsv=True)
        normals = get_normals(ros_cluster)
        nhists = compute_normal_histograms(normals)
        feature = np.concatenate((chists, nhists))

        # Make the prediction
        prediction = clf.predict(scaler.transform(feature.reshape(1, -1)))
        label = encoder.inverse_transform(prediction)[0]
        detected_objects_labels.append(label)

        # Publish a label into RViz
        label_pos = list(white_cloud[pts_list[0]])
        label_pos[2] += .3        # .4
        object_markers_pub.publish(make_label(label, label_pos, index))

        # Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = ros_cluster
        detected_objects.append(do)

    rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))

    # Publish the list of detected objects
    detected_objects_pub.publish(detected_objects)

    # Suggested location for where to invoke your pr2_mover() function within pcl_callback()
    # Could add some logic to determine whether or not your object detections are robust
    # before calling pr2_mover()
    try:
        pr2_mover(detected_objects, detected_objects_labels)
    except rospy.ROSInterruptException:
        pass

# function to load parameters and request PickPlace service
def pr2_mover(object_list, object_labels):

    # Initialize variables
    test_scene_num = Int32()
    test_scene_num.data = 1
    object_name = String()
    arm = String()
    pick_pose = Pose()
    place_pose = Pose()
    dict_list = []

    # Get/Read parameters
    object_list_param = rospy.get_param('/object_list')
    place_pose_param = rospy.get_param('/dropbox')

    # Parse parameters into individual variables
    dropbox_name = []
    dropbox_group = []
    dropbox_position = []
    for i in range(len(place_pose_param)):
        dropbox_name.append(place_pose_param[i]['name'])
        dropbox_group.append(place_pose_param[i]['group'])
        dropbox_position.append(place_pose_param[i]['position'])

    # TODO: Rotate PR2 in place to capture side tables for the collision map

    # Loop through the pick list
    # Get the PointCloud for a given object and obtain it's centroid
    for counter, object in enumerate(object_list_param):

        # Parse parameters into individual variables
        object_group = object_list_param[counter]['group']
        object_name.data = object_list_param[counter]['name']

        # Check if the object we want to pick up is in the list of detected objects
        if object_name.data in object_labels:
            object_index = object_labels.index(object_name.data)
            # Calculate the centroid
            points_arr = ros_to_pcl(object_list[object_index].cloud).to_array()
            centroid_tuple = np.mean(points_arr, axis=0)[:3]

            # Pick pose
            pick_pose.position.x = np.asscalar(centroid_tuple[0])
            pick_pose.position.y = np.asscalar(centroid_tuple[1])
            pick_pose.position.z = np.asscalar(centroid_tuple[2])
            pick_pose.orientation.x = 0
            pick_pose.orientation.y = 0
            pick_pose.orientation.z = 0
            pick_pose.orientation.w = 0

            # Create 'place_pose' for the object
            i = dropbox_group.index(object_group)
            place_pose.position.x = dropbox_position[i][0]
            place_pose.position.y = dropbox_position[i][1]
            place_pose.position.z = dropbox_position[i][2]
            place_pose.orientation.x = 0
            place_pose.orientation.y = 0
            place_pose.orientation.z = 0
            place_pose.orientation.w = 0
            # Create arm name
            arm.data = dropbox_name[i]

            # Create a list of dictionaries (made with make_yaml_dict()) for later output to yaml format
            # Populate various ROS messages
            yaml_dict = make_yaml_dict(test_scene_num, arm, object_name, pick_pose, place_pose)
            dict_list.append(yaml_dict)

            # Wait for 'pick_place_routine' service to come up
            # rospy.wait_for_service('pick_place_routine')
            #
            # try:
            #     pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)
            #
            #     # TODO: Insert your message variables to be sent as a service request
            #     resp = pick_place_routine(TEST_SCENE_NUM, OBJECT_NAME, WHICH_ARM, PICK_POSE, PLACE_POSE)
            #
            #     print ("Response: ",resp.success)
            #
            # except rospy.ServiceException, e:
            #     print "Service call failed: %s"%e

    # Output your request parameters into output yaml file
    send_to_yaml('out.yaml', dict_list)

    print 'Iteration end'

if __name__ == '__main__':

    # ROS node initialization
    rospy.init_node('object_recognition', anonymous=True)

    # Create Subscribers
    pcl_sub = rospy.Subscriber("/pr2/world/points", pc2.PointCloud2, pcl_callback, queue_size=1)

    # Create Publishers
    object_markers_pub = rospy.Publisher("/object_markers", Marker, queue_size=1)
    detected_objects_pub = rospy.Publisher("/detected_objects", DetectedObjectsArray, queue_size=1)
    pcl_objects_pub = rospy.Publisher("/pcl_objects", PointCloud2, queue_size=1)
    pcl_table_pub = rospy.Publisher("/pcl_table", PointCloud2, queue_size=1)
    pcl_cluster_cloud_pub = rospy.Publisher("/pcl_cluster", PointCloud2, queue_size=1)

    # Load Model From disk
    model = pickle.load(open('model.sav', 'rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']

    # Initialize color_list
    get_color_list.color_list = []

    print('Running...')

    # Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()
