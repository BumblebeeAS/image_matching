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

To obtain a pose from the output, run PnP on the 2D–3D correspondences (e.g., see [pose_estimator/points_pose_estimator_node](https://github.com/BumblebeeAS/pose_estimator/blob/main/pose_estimator/points_pose_estimator_node.py)). We don't run PnP here since these raw correspondences can be combined with other detections downstream for a better estimate.

Use XFeat for general, upright camera matching. For matching between images with large orientation differences (e.g., drone imagery), try SIFT-FLANN or [DALF](https://github.com/verlab/DALF_CVPR_2023).

See [this commit](https://github.com/BumblebeeAS/image_matching/commit/72e45e7c73e4010efcecb51d2a4896534290abae) for the old image matchers that the seniors used.

## References

(Outdated) https://github.com/Shiaoming/Python-VO
