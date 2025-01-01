import tensorflow as tf
from waymo_open_dataset.protos import scenario_pb2
import pickle
from matplotlib import pyplot as plt
import numpy as np
import tqdm
import random

# with open('scenario_id_map.pkl', 'rb') as f:
#     scenario_id_map = pickle.load(f)
# print(scenario_id_map)

FILENAME = "/home/zhengzhilong/data/womd/scenario/training/training.tfrecord-00000-of-01000"
# FILENAME = "/home/zhengzhilong/data/womd/scenario/training/modified-training.tfrecord-00000-of-01000"
raw_dataset = tf.data.TFRecordDataset([FILENAME])
scenario_id_map = {}
# proto = scenario_pb2.Scenario()

# raw_records = [r for r in raw_dataset.take(3)]
plt.figure(dpi=1000)
# plt.axis('equal')
# plt.xlim(3300, 3500)
# plt.ylim(1500, 1600)
flag = True
for raw_record in raw_dataset.take(1):
    proto_string = raw_record.numpy()
    proto = scenario_pb2.Scenario()
    proto.ParseFromString(proto_string)
    scenario_id_map[int(proto.scenario_id, 16)] = FILENAME.split('/')[-1]
    map_features = proto.map_features
    map_feature_dict = {map_feature.id: map_feature for map_feature in map_features}
    while True:
      random_lane_id = random.choice(list(map_feature_dict.keys()))
      if map_feature_dict[random_lane_id].lane.polyline:
          break
    random_lane_id = 637
    for map_feature in tqdm.tqdm(map_features):
        if map_feature.lane.polyline:
            xys = [(point.x, point.y) for point in map_feature.lane.polyline]
            xys = np.array(xys)
            if map_feature.id == random_lane_id:
                plt.plot(xys[:, 0], xys[:, 1], c='red', linewidth=0.5)
                flag = False
                for neighbor in map_feature.lane.left_neighbors:
                    neighbor_feature = map_feature_dict[neighbor.feature_id]
                    xys = [(point.x, point.y) for point in neighbor_feature.lane.polyline]
                    xys = np.array(xys)
                    plt.plot(xys[neighbor.neighbor_start_index:neighbor.neighbor_end_index, 0], xys[neighbor.neighbor_start_index:neighbor.neighbor_end_index, 1], c='blue', linewidth=0.5, zorder=1)
                for neighbor in map_feature.lane.right_neighbors:
                    neighbor_feature = map_feature_dict[neighbor.feature_id]
                    xys = [(point.x, point.y) for point in neighbor_feature.lane.polyline]
                    xys = np.array(xys)
                    plt.plot(xys[neighbor.neighbor_start_index:neighbor.neighbor_end_index, 0], xys[neighbor.neighbor_start_index:neighbor.neighbor_end_index, 1], c='green', linewidth=0.5, zorder=1)
                for id in map_feature.lane.entry_lanes:
                    entry_feature = map_feature_dict[id]
                    xys = [(point.x, point.y) for point in entry_feature.lane.polyline]
                    xys = np.array(xys)
                    plt.plot(xys[:, 0], xys[:, 1], c='yellow', linewidth=0.5, zorder=1)
                for id in map_feature.lane.exit_lanes:
                    exit_feature = map_feature_dict[id]
                    xys = [(point.x, point.y) for point in exit_feature.lane.polyline]
                    xys = np.array(xys)
                    plt.plot(xys[:, 0], xys[:, 1], c='orange', linewidth=0.5, zorder=1)
                    for neighbor in exit_feature.lane.left_neighbors:
                        neighbor_feature = map_feature_dict[neighbor.feature_id]
                        xys = [(point.x, point.y) for point in neighbor_feature.lane.polyline]
                        xys = np.array(xys)
                        plt.plot(xys[neighbor.neighbor_start_index:neighbor.neighbor_end_index, 0], xys[neighbor.neighbor_start_index:neighbor.neighbor_end_index, 1], c='blue', linewidth=0.5, zorder=1)
                    for neighbor in exit_feature.lane.right_neighbors:
                        neighbor_feature = map_feature_dict[neighbor.feature_id]
                        xys = [(point.x, point.y) for point in neighbor_feature.lane.polyline]
                        xys = np.array(xys)
                        plt.plot(xys[neighbor.neighbor_start_index:neighbor.neighbor_end_index, 0], xys[neighbor.neighbor_start_index:neighbor.neighbor_end_index, 1], c='green', linewidth=0.5, zorder=1)
            else:
                plt.plot(xys[:, 0], xys[:, 1], c='grey', linewidth=0.5, zorder=0)
        elif map_feature.road_edge.polyline:
            xys = [(point.x, point.y) for point in map_feature.road_edge.polyline]
            xys = np.array(xys)
            plt.plot(xys[:, 0], xys[:, 1], c='black', linewidth=0.5, zorder=0)
print(random_lane_id)
plt.savefig(f'/home/zhengzhilong/data/womd/scenario/training/map_{random_lane_id}.png')


# modified_dataset = tf.data.Dataset.from_tensor_slices(raw_records[::-1])
# filename = "/home/zhengzhilong/data/womd/scenario/training/modified-training.tfrecord-00000-of-01000"
# writer = tf.data.experimental.TFRecordWriter(filename)
# writer.write(modified_dataset)

# print(raw_records)
    # print(int(proto.scenario_id, 16))
#   print(proto)
#   break
# save scenario_id_map
# with open('scenario_id_map.pkl', 'wb') as f:
#     pickle.dump(scenario_id_map, f)