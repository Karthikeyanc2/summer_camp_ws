import rosbag


bag = rosbag.Bag("/home/schenker2/bags/summer_camp/summer_camp_3.bag")

for topic, msg, t in bag.read_messages(topics=["/tf", "/vehicle_marie/adma_data", "/vehicle_marie/vehicle_marker", "/vehicle_marie/velodyne_points"]):
    if topic == "/tf":
        if 'issaak' in msg.transforms[0].child_frame_id:
            continue
        print(msg.transforms[0].transform.translation)
    if topic == "/vehicle_marie/vehicle_marker":
        print(msg.pose.position)
    elif topic == "/vehicle_marie/adma_data":
        print(msg.pose_cg.position)
        if msg.pose_cg.position.x != -97.45193624662637:
            b = 1
    elif topic == "/vehicle_marie/velodyne_points":
        print(msg.header.stamp)
    a = 1