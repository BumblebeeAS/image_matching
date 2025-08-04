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
ros2 run image_matching simple_matcher_node
```

In another terminal:

```bash
source install/setup.bash
ros2 service call /image_matching/toggle_template bb_perception_msgs/srv/IMPoseEstimatorToggleTemplate "template_name: 'Task04_Tagging_01.png'
camera_frame_id: 'auv4/front_cam_optical'
enable: true"
```

See `image_matching/image` for visualization. Point correspondences are published at `image_matching/point_correspondences`.

## Notes

Use XFeat for general, upright camera matching. For matching between images with large orientation differences (e.g., drone imagery), try SIFT-FLANN.

## References

(Outdated) https://github.com/Shiaoming/Python-VO
