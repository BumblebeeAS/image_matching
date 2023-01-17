#!/usr/bin/env python3
import numpy as np
import rospy
import cv2
import os
from pathlib import Path
from rospkg import RosPack

from detector import BasicFeatureMatcher
from cv_bridge import CvBridge, CvBridgeError
from feature_matcher.keypoints_match_producer import \
    get_keypoints_match_producer

class annotator(BasicFeatureMatcher):

    def convertToRect(self, pts, h, w):
        pts = np.reshape(pts,(4,2))
        max_pt = np.amax(pts, axis=0)
        min_pt = np.amin(pts, axis=0)
        #Limit checking
        #if (max_pt > (h,w)):
        #        max_pt = np.float32([h,w])
        #if (min_pt < (0,0)):
        #    min_pt = np.float32([0,0])
        return np.float32([max_pt[0], max_pt[1], max_pt[0], min_pt[1], min_pt[0], min_pt[1], min_pt[0], max_pt[1]]).reshape(4,1,2)

    def cropped_image_callback(self, img_msg, detected_objects=None, debug=False):
        rospy.logdebug_throttle(10, f"Received image {img_msg.header.seq}")
        try:
            img = self.bridge.compressed_imgmsg_to_cv2(img_msg, "bgr8")
        except CvBridgeError as e:
            print(e)

        detected_object = None
        if self.CROP_IMAGES and detected_objects is not None:
            if any([x.name == self.template for x in detected_objects.detected]):
                detected_object = sorted(detected_objects.detected, key=lambda x: x.extra[0], reverse=True)[0]

        if detected_object is not None:
            PADDING = 10
            cx, cy, w, h = detected_object.centre_x, detected_object.centre_y, detected_object.bbox_width, detected_object.bbox_height
            x, y = int(cx - w / 2), int(cy - h / 2)
            lxtyrxby = (max(0, x-PADDING), max(0, y-PADDING), min(img.shape[1], x+w+PADDING), min(img.shape[0], y+h+PADDING))
        else: 
            lxtyrxby = None


        kp1, kp2 = self.image_match_producer.process_image(img, self.template, lxtyrxby = lxtyrxby,  debug=True) #kp1 -> template, kp2 -> img
        src_pts = np.float32([kp1.keypoints]).reshape(-1,1,2) 
        dst_pts = np.float32([kp2.keypoints]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        h,w = self.template_img.shape[:2]

        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        
        if(M is not None):
            dst = cv2.perspectiveTransform(pts,M)
            dst = self.convertToRect(dst, h, w)

            img2 = cv2.polylines(img, [np.int32(dst)], True, (0,255,0),3, cv2.LINE_AA)

            #Write to path
            annotation_path = os.path.abspath(Path(RosPack().get_path("image_matching"))/"images")

            if not (os.path.exists(annotation_path)):
                os.mkdir(annotation_path)

            if not cv2.imwrite("{}/id_{}.jpg".format(annotation_path, img_msg.header.seq), img):
                raise Exception("Could not write image")

            with open(os.path.join(annotation_path,"label.txt"), "a") as f:
                temp_pts = np.reshape(dst,(4,2))
                min_x, min_y = np.amin(temp_pts, axis=0)
                w,h = abs(np.amax(temp_pts, axis=0) - (min_x, min_y))
                f.write("id_{} {} {} {} {} {}\n".format(img_msg.header.seq, min_x, min_y, w, h, template))
                f.close()

if __name__ == "__main__":
    rospy.init_node("basic_bounding_box", anonymous=True, log_level=rospy.DEBUG)
    camera_topic = rospy.get_param("~camera_topic", "/auv4/front_cam/image_color/compressed")
    visualization_topic = rospy.get_param("~visualization_topic", "/visualization")
    template = rospy.get_param("~template", "Badge")
    template_path = rospy.get_param("~template_path", os.path.abspath(Path(RosPack().get_path("image_matching"))/"templates"/f"{template}.jpeg"))
    detected_objects_topic = rospy.get_param("~detected_objects_topic", None)
    detector = annotator(camera_topic, 
                        visualization_topic,
                        template,
                        template_path,
                        detected_objects_topic)
    rospy.spin()

    
    #template image
