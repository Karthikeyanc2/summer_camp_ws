import rosbag

input_bag_file = "/home/schenker2/bags/summer_camp/tuesday/s3.bag"
required_topics = ["/vehicle_issaak/data_raw", "/vehicle_marie/data_raw", "/vehicle_marie/velodyne/data_raw",
                   "/vehicle_marie/velodyne/tele_raw"]
output_bag_file = input_bag_file.replace('.bag', '_raw.bag')

ip_bag = rosbag.Bag(input_bag_file)
op_bag = rosbag.Bag(output_bag_file, 'w')

for topic, msg, t in ip_bag.read_messages(topics=required_topics):
    op_bag.write(topic, msg, t)

ip_bag.close()
op_bag.close()
