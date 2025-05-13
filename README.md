# Image matching

## System

- ROS2 Humble
- Pytorch
- Numpy
- OpenCV
- CUDA

## Quickstart

In one terminal:

```bash
source install/setup.bash
ros2 launch image_matching simple_pose_estimator.launch.py
```

In another terminal:

```bash
source install/setup.bash
ros2 service call /auv4/image_matching/toggle_template bb_msgs/srv/IMPoseEstimatorToggleTemplate "template_name: 'Task04_Tagging_01.png'
camera_frame_id: 'auv4/front_cam_optical'
enabled: true"
```

See `/impose_dev_vis/compressed` for visualization.

Credits:
https://github.com/Shiaoming/Python-VO
