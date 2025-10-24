import sys
import os

# Add the local lib path to import the pybind11 module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "lib")))

import ros2_pybindings
import traceback

try:
    import ros2_pybindings
    from cv_bridge import CvBridge
except:
    # print(traceback.format_exc())
    print("Check ROS2 is installed and sourced, and ros2_pybindings was correctly built")
    sys.exit(1)


def main():
    bag_path = "/home/luigi/Work/datasets/rgbd_datasets/tum/rgbd_dataset_freiburg1_room/rgbd_dataset_freiburg1_room.ros2.bag"
    topics = ["/camera/rgb/image_color", "/camera/depth/image"]
    queue_size = 100
    slop = 0.05

    print(f"Loading bag: {bag_path}")
    print(f"Topics: {topics}")
    reader = ros2_pybindings.Ros2BagSyncReaderATS(bag_path, topics, queue_size, slop)

    count = 0
    while not reader.is_eof():
        result = reader.read_step()
        if result is None:
            # print("No synchronized messages yet")
            continue

        timestamp, msg_dict = result
        if len(msg_dict) < len(topics):
            # print("Skipped incomplete sync")
            continue

        print(f"[{count}] Timestamp: {timestamp:.6f}")
        for topic in topics:
            msg = msg_dict.get(topic)
            if msg:
                t = msg.header.stamp
                print(
                    f"  - Topic: {topic}, Frame: {msg.header.frame_id}, Time: {t.sec}.{str(t.nanosec).zfill(9)}"
                )
        count += 1

    print(f"\n✅ Finished reading {count} synchronized steps")


if __name__ == "__main__":
    main()
