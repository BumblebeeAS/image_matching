from distutils.core import setup

setup(
    name="image_matching",
    version="0.0.0",
    packages=["feature_matcher", "pose_estimator"],
    package_dir={"": "src"},
)
