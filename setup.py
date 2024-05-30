from setuptools import setup, find_packages
import os
from glob import glob
package_name = 'image_matching'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name] + find_packages(where="./src"),
    package_dir={"":"./src",
                 "image_matching": "./image_matching"},
    # packages=[
    #     package_name,
    #     "feature_matcher",
    #     "feature_matcher.keypoint_matcher",
    #     "feature_matcher.keypoint_producer",
    #     "feature_matcher.models",
    #     "pose_estimator", 
    #     "utils"],
    # package_dir={"feature_matcher": "src/feature_matcher", 
    #              "pose_estimator": "src/pose_estimator",
    #              "keypoint_matcher": "src/feature_matcher/keypoint_matcher",
    #              "keypoint_producer": "src/feature_matcher/keypoint_producer",
    #              "utils": "src/utils"
    #              },
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        (f"share/{package_name}/models/SuperGluePretrainedNetwork/models/weights",
            ["src/feature_matcher/models/SuperGluePretrainedNetwork/models/weights/superpoint_v1.pth"]),
        (f"share/{package_name}/templates",
            glob('templates/*.json') + glob('templates/*.png') + glob('templates/*.jpg') + glob('templates/*.jpeg')),
        (f"share/{package_name}/models/accelerated_features/weights/", ["src/feature_matcher/models/accelerated_features/weights/xfeat.pt"]),
        (f"share/{package_name}/models/accelerated_features/modules", glob("src/feature_matcher/models/accelerated_features/modules/*.py"))
    ],
    install_requires=['setuptools',],
    zip_safe=True,
    maintainer='b3nguin',
    maintainer_email='koh.benjamin@u.nus.edu',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'detector = image_matching.detector:main',
            'pose_estimator = image_matching.pose_estimator_node:main',
            'test_xfeat = image_matching.test_xfeat:main',
        ],
    },
)
