import numpy as np
from feature_matcher.keypoints_match_producer import Keypoints
from feature_matcher.keypoint_producer.xfeat import XFeatKeypointProducer
from feature_matcher.keypoint_matcher.xfeat import XFeatKeypointMatcher
import os
from pathlib import Path
from ament_index_python import get_package_share_directory
import cv2
from feature_matcher.tools import plot_matches

test_folder_path = os.path.join(get_package_share_directory("image_matching"), "test_images")
template_path = os.path.abspath(
    Path(os.path.realpath(__file__)).parents[1]
    / "benchmark"
)  # noqa E402

def process_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error reading {image_path}")
        return None
    
    print(f"Successfully read {image_path}")
    return image

def ensure_dir_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path, exist_ok=True)
        print(f"Created directory: {directory_path}")
    else:
        print(f"Directory already exists: {directory_path}")

def main():
    producer = XFeatKeypointProducer()
    matcher = XFeatKeypointMatcher()
    for class_name in os.listdir(test_folder_path):
        class_folder_path = os.path.join(test_folder_path, class_name)
        
        template_image_path = os.path.join(template_path, f"{class_name}.png")
        template_image = cv2.imread(template_image_path)
        
        if template_image is None:
            print(f"Failed to read template image for {class_name}")
            continue

        template_keypoints = producer(template_image)

        if os.path.isdir(class_folder_path):
            print(f"Processing {class_name}")
            out = 0
            for image_name in os.listdir(class_folder_path):
                image_path = os.path.join(class_folder_path, image_name)

                image = process_image(image_path)

                if image is not None:
                    output_dir = os.path.join(get_package_share_directory("image_matching"), f"output/xfeat/{class_name}")
                    ensure_dir_exists(output_dir)
                    output_image_path = os.path.join(output_dir, f"{out}.jpg")

                    test_image_keypoints = producer(image)

                    match_kp1, match_kp2 = matcher(template_keypoints, test_image_keypoints)
                    output = plot_matches(template_image, image, match_kp1.keypoints, match_kp2.keypoints)
                    if cv2.imwrite(output_image_path, output):
                        print(f"Matched output in {output_image_path}")
                    else:
                        print(f"Failed to write {output_image_path}")
                
                out += 1

        print(f"Processed {class_name}")

    print(f"Processed all test images")

if __name__ == "__main__":
    main()
