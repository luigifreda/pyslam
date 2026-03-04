import pyrealsense2 as rs
import numpy as np
import cv2
import os
import math

#Choose which files you want to create from Realsense bag.
GENERATE_IMG   = False
GENERATE_ASSOC = False
GENERATE_GT    = True


base = "/home/albincederberg/Videos/"
bag = base+"Bags/LoopTest.bag" #File path for recorded realsense bag
dir = base+"LoopTest" #Path of the folder that will contain the folders:[RGB], [Depth] and the files: associations.txt, groundtruth.txt


rgb_dir = dir+"/rgb" #[RGB]- image folder
depth_dir = dir+"/depth" #[Depth]-image folder
os.makedirs(rgb_dir, exist_ok=True)
os.makedirs(depth_dir, exist_ok=True)

#Fattar inte dessa än /Albin
pipeline = rs.pipeline()
config = rs.config()
config.enable_device_from_file(bag, repeat_playback=False)
profile = pipeline.start(config)
align = rs.align(rs.stream.color)

last_accel = None
last_accel_ts = None

rgb_file = open(dir+"/rgb.txt","w")
depth_file = open(dir+"/depth.txt","w")


#Association params
assoc_file = dir+"/associations.txt" #associations
rgb_files = sorted(os.listdir(rgb_dir))
depth_files = sorted(os.listdir(depth_dir))


#GT-params
gt_out =  dir + "/groundtruth.txt"
lidar_data=  base + "/LidarData/"+ "LoopTest"


################IMAGES########################


if GENERATE_IMG:
    get_first_t = True
    print("Extracting... this will take the video duration")
    while True:
        try:
            frames = pipeline.wait_for_frames()
        except:
            GENERATE_IMG = False #image extraction finished, stop loop
            break

        frames = align.process(frames)

        depth = frames.get_depth_frame()
        color = frames.get_color_frame()

        if get_first_t:
            start_t = color.get_timestamp()/1000.0
            latest_printed_t = 0
            get_first_t = False


        t = color.get_timestamp()/1000.0
        if int(t-start_t)%50 == 0 and int(t-start_t) != latest_printed_t:
            print(f"{int(t-start_t)} Seconds of video extracted...")
            latest_printed_t = int(t-start_t)

        rgb_img = np.asanyarray(color.get_data())
        depth_img = np.asanyarray(depth.get_data())

        # RealSense returns color frames in RGB order, but OpenCV assumes BGR.
        bgr_img = rgb_img[..., ::-1] #Convert RGB to BGR for OpenCV

        rgb_name = f"rgb/{t:.6f}.png"
        depth_name = f"depth/{t:.6f}.png"

        cv2.imwrite(dir+"/"+rgb_name, bgr_img)
        cv2.imwrite(dir+"/"+depth_name, depth_img)

        rgb_file.write(f"{t:.6f} {rgb_name}\n")
        depth_file.write(f"{t:.6f} {depth_name}\n")
    print(f"{int(t-start_t)} RGB and Depth images extracted.")


pipeline.stop()

rgb_file.close()
depth_file.close()




################ASSOCIATIOS########################
rgb_files = sorted(os.listdir(rgb_dir))
depth_files = sorted(os.listdir(depth_dir))
def timestamp(name):
    return float(os.path.splitext(name)[0])

if GENERATE_ASSOC:
    i = 0
    j = 0

    print("Associations will be written to ",assoc_file)
    with open(assoc_file, "w") as f:
        while i < len(rgb_files) and j < len(depth_files):

            t_rgb = timestamp(rgb_files[i])
            t_depth = timestamp(depth_files[j])

            diff = abs(t_rgb - t_depth)

            # max 20 ms difference
            if diff < 0.02:
                f.write(f"{t_rgb:.6f} rgb/{rgb_files[i]} {t_depth:.6f} depth/{depth_files[j]}\n")
                i += 1
                j += 1

            elif t_rgb < t_depth:
                i += 1
            else:
                j += 1
    print("Associations file created!")


################GT########################

if GENERATE_GT:
    print("Ground truth will be written to ",gt_out)
    f = open(lidar_data, "r", encoding="utf-8")
    data = f.read()
    lines = data.split("\n")
    lines.pop(0)

    states = []
    text = ""
    for line in lines:
        if "state" in line:
            vals = line.split(" ")
            qz = str(math.sin(float(vals[4])/2))                                        
            qw = str(math.cos(float(vals[4])/2))
            # TUM FORMAT [t,x,y,z,qx,qy,qz,qw]
            # Does not seem to impact SLAM but has tobe on CV coordinates for VO to work.
            #text += f"{vals[1]} {vals[2]} {vals[3]} 0 0 0 {qz} {qw}\n" # #Robot coordinates with z up, x forward, y left]
            text += f"{vals[1]} -{vals[3]} 0 -{vals[2]} 0 0 {qz} {qw}\n"  #CV coordinates with z forward, x right, y down]

    with open(gt_out, "w") as f:
        f.write(text)
        print("Groundtruth file created!")