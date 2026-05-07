import os
from glob import glob

from setuptools import find_packages, setup

package_name = "image_matching"

setup(
    name=package_name,
    version="0.0.0",
    packages=[package_name] + find_packages(where="./src"),
    package_dir={
        "feature_matcher": "src/feature_matcher",
        "keypoint_matcher": "src/feature_matcher/keypoint_matcher",
        "keypoint_producer": "src/feature_matcher/keypoint_producer",
        "utils": "src/utils",
    },
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        ("share/" + package_name, ["package.xml"]),
        (
            os.path.join("share", package_name, "launch"),
            glob(os.path.join("launch", "*launch.[pxy][yma]*")),
        ),
        (
            os.path.join("share", package_name, "cfg"),
            glob(os.path.join("cfg", "*.yaml")),
        ),
        # TODO: Make feature_matcher and pose_estimator proper packages and use the following
        # (
        #     os.path.join("share", package_name, "src"),
        #     glob("src/**/*", recursive=True),
        # ),
        # (
        #     os.path.join(
        #         "share",
        #         package_name,
        #         "src/feature_matcher/models/accelerated_features/weights",
        #     ),
        #     glob(
        #         os.path.join(
        #             "src/feature_matcher/models/accelerated_features/weights", "*.pt"
        #         )
        #     ),
        # ),
        (
            os.path.join("share", package_name, "templates"),
            glob(os.path.join("templates", "*.json"), recursive=True)
            + glob(os.path.join("templates", "*.png"), recursive=True)
            + glob(os.path.join("templates", "*.jpg"), recursive=True),
        ),
        # (
        #     os.path.join("share", package_name, "templates", "robosub25"),
        #     glob(
        #         os.path.join("templates", "robosub25", "*.json"),
        #         recursive=True,
        #     )
        #     + glob(
        #         os.path.join("templates", "robosub25", "*.png"), recursive=True
        #     )
        #     + glob(
        #         os.path.join("templates", "robosub25", "*.jpg"), recursive=True
        #     ),
        # ),
        (
            os.path.join("share", package_name, "templates", "robosub26"),
            glob(
                os.path.join("templates", "robosub26", "*.json"),
                recursive=True,
            )
            + glob(
                os.path.join("templates", "robosub26", "*.png"), recursive=True
            )
            + glob(
                os.path.join("templates", "robosub26", "*.jpg"), recursive=True
            ),
        ),
    ],
    install_requires=[
        "setuptools",
    ],
    zip_safe=True,
    maintainer="b3nguin",
    maintainer_email="koh.benjamin@u.nus.edu",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "detector = image_matching.detector:main",
            "test_xfeat = image_matching.test_xfeat:main",
            "xfeat_output = image_matching.xfeat_output:main",
            "simple_matcher_node = image_matching.simple_matcher_node:main",
        ],
    },
)
