import rosbag

input_bag_file = "/home/schenker2/bags/summer_camp/tuesday/s3.bag"
input_bag_file = "/home/schenker2/isaak_rotating_marie.bag"
input_bag_file = "/home/schenker2/bags/summer_camp/wednesday/s1_v2.bag"
input_bag_file = "/home/schenker2/karthik/summer_camp_ws/src/ttc_based_critical_situation_identifier/bags/test.bag"

input_bag_file = "/home/schenker2/bags/summer_camp/thursday/s1_v1.bag"

required_topics = ["/vehicle_isaak/data_raw", "/vehicle_issaak/data_raw", "/vehicle_marie/data_raw", "/vehicle_marie/velodyne/data_raw",
                   "/vehicle_marie/velodyne/tele_raw"]
output_bag_file = input_bag_file.replace('.bag', '_raw.bag')

ip_bag = rosbag.Bag(input_bag_file)
op_bag = rosbag.Bag(output_bag_file, 'w')

for topic, msg, t in ip_bag.read_messages(topics=required_topics):
    topic = topic.replace('issaak', 'isaak')
    op_bag.write(topic, msg, t)

ip_bag.close()
op_bag.close()
