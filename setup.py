from setuptools import setup

package_name = 'image_matching'

def generate_data_files(share_path, dir):
    data_files = []

    for path, _, files in os.walk(dir):
        list_entry = (share_path + path, [os.path.join(path, f) for f in files if not f.startswith('.')])
        data_files.append(list_entry)

    return data_files

setup(
    name=package_name,
    version='0.0.0',
    packages=[
        package_name,
        "src/feature_matcher", 
        "src/pose_estimator", 
        "src/utils"],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
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
        ],
    },
)
