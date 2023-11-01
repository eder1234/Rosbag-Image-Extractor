## Rosbag Image Extractor

This script is designed to extract color and depth images from a rosbag file and save them as `.png` images in designated folders. Additionally, it captures essential camera calibration parameters and stores them as text files.

### Prerequisites:

- Python 2.7
- ROS (Robot Operating System) with `rosbag` and `rospy` packages installed
- OpenCV for Python 2 (`cv2`)
- `cv_bridge` for converting between ROS image messages and OpenCV images

### Usage:

To run the script, use the following command:

```
python2.7 extract_images_from_rosbag.py /path/to/your/rosbag/file.bag /path/to/output/folder/
```

Replace `/path/to/your/rosbag/file.bag` with the path to your rosbag file and `/path/to/output/folder/` with the directory where you wish the images and camera info text files to be saved.

### Features:

- **Image Extraction**: The script looks for topics `/device_0/sensor_1/Color_0/image/data` and `/device_0/sensor_0/Depth_0/image/data` in the rosbag to extract color and depth images respectively.
- **Camera Info Extraction**: It also captures camera calibration parameters such as camera matrix, distortion coefficients, rectification matrix, and projection matrix from topics `/device_0/sensor_1/Color_0/camera_info` and `/device_0/sensor_0/Depth_0/camera_info`.
- **Directory Organization**: Extracted images are saved in separate folders named `color` and `depth` within the specified output directory. Camera information for both color and depth sensors is stored in respective text files named `color_camera_info.txt` and `depth_camera_info.txt`.

### Note:

Ensure that the script has executable permissions. If not, use the following command:

```
chmod +x extract_images_from_rosbag.py
```
All the data was collected from an Intel RealSense D415.
