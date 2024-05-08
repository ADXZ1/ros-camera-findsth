#!/usr/bin/python3
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import os


class ImageProcessor:
    def __init__(self):
        self.node_name = "image_processor"

        rospy.init_node(self.node_name)

        self.bridge = CvBridge()

        self.image_sub = rospy.Subscriber("/camera/image_raw", Image, self.image_callback)


    def image_callback(self, data):

        try:
            # 将ROS图像消息转换为OpenCV图像
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        # 将图像转换为 HSV 颜色空间
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # 定义绿色的 HSV 范围
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([90, 255, 255])

        # 创建绿色的掩码
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # 保留绿色部分
        img_green = cv2.bitwise_and(cv_image,cv_image, mask=mask)
        img = cv2.cvtColor(img_green, cv2.COLOR_BGR2GRAY)

        # 使用阈值分割绿色部分
        _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        #去除噪点
        kernel = np.ones((5, 5), np.uint8)
        denoised_thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        denoised_thresh = cv2.morphologyEx(denoised_thresh, cv2.MORPH_OPEN, kernel)

        # 查找轮廓
        contours, _ = cv2.findContours(denoised_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # 为每个轮廓绘制矩形框
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x_centre = x + w / 2
            y_centre = y + h / 2
            cv2.circle(cv_image,(int(x_centre),int(y_centre)),1,(0,0,255),2)
            cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            print({"x": x_centre, "y": y_centre})

        cv2.imshow("Green Image", cv_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    image_processor = ImageProcessor()
    rospy.spin()
    cv2.destroyAllWindows()
