"""
This function allows us to snap/create new template images for objects of interest
using conventional ML. It is a service callback. Does not update the templates.json
Please manually update the templates.json file to use the newly taken image.
"""
def register_template_cb(
    self, req: IMPoseEstimatorRegisterTemplate.Request, res
):
    compressed_image_topic_name = req.image_topic_name
    detected_objects_topic_name = req.detected_objects_topic_name
    object_name = req.object_name
    detected_object = None
    try:
        if detected_objects_topic_name != "" and object_name != "":
            for i in range(3):
                valid, detected_objects = wait_for_message(
                    DetectedObjects,
                    self,
                    detected_objects_topic_name,
                    time_to_wait=2,
                )
                if not valid:
                    continue
                if any(
                    [
                        x.name == object_name
                        for x in detected_objects.detected
                    ]
                ):
                    detected_object = sorted(
                        detected_objects.detected,
                        key=lambda x: x.extra[0],
                        reverse=True,
                    )[0]
                    break
        valid, img = wait_for_message(
            CompressedImage,
            self,
            compressed_image_topic_name,
            time_to_wait=2,
        )
        if not valid:
            raise ValueError("failed to get message")
        cv2_img: np.ndarray = self.bridge.compressed_imgmsg_to_cv2(img)
        if detected_object is not None:
            PADDING = 10
            cx, cy, w, h = (
                detected_object.centre_x,
                detected_object.centre_y,
                detected_object.width,
                detected_object.height,
            )
            x, y = int(cx - w / 2), int(cy - h / 2)
            cv2_img = cv2_img[
                y - PADDING : y + h + PADDING,
                x - PADDING : x + w + PADDING,
                :,
            ]
        cv2.imwrite(
            os.path.join(
                self.templates_dir,
                f"{req.template_name}.{self.get_clock().now().secs}.jpg",
            ),
            cv2_img,
        )
        if req.template_name in self.templates:
            self.get_logger().info(
                "Replacing existing template %s", req.template_name
            )
        self.register_template(
            cv2_img,
            req.template_name,
            (req.width, req.height),
            object_name,
            (0, 0),
        )
        res.success = True
        return res
    except Exception as e:
        res.success = False
        res.error_message = str(e)
        return res

@staticmethod
def equalize_green_blue(img):
    b, g, r = cv2.split(img)

    g_eq = cv2.equalizeHist(g)
    b_eq = cv2.equalizeHist(b)

    img_eq = cv2.merge((b_eq, g_eq, r))

    return img_eq
