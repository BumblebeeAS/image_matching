## Migration in progress for Humble Branch

# SuperGlue ROS
ROS wrapper for SuperGlue and SuperPoint models

 - SuperGlue: [paper](https://arxiv.org/abs/1911.11763)
 - SuperPoint: [paper](https://arxiv.org/abs/1712.07629)

You can find **the utilization of this module** in [this](https://github.com/KopanevPavel/runbot_custom_localization) repository in the frontend part of the visual-inertial SLAM system

## System
 - ROS2 Humble (with python3 support)
 - Pytorch
 - Numpy
 - OpenCV
 - CUDA (highly recommended)

## Quickstart

In one terminal:

```bash
source install/setup.bash
ros2 launch image_matching pose_estimator.launch.py
```

In another terminal:

```bash
source install/setup.bash
ros2 service call /impose_toggle_template bb_msgs/srv/IMPoseEstimatorToggleTemplate "template_name: '2023_torpedo_big'
camera_frame_id: 'auv4/bot_cam_optical'
enabled: true"
```

See `/impose_dev_vis/compressed` for visualization.

Credits:
https://github.com/Shiaoming/Python-VO
