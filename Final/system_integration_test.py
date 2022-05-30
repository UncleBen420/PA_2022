#!/usr/bin/env python3
from ProducerConsumer import EventProducerKafka
from ProducerConsumer import EventConsumerKafka
from ShopLiftingDetector import ShopLiftingDetector
import cv2
import time
import numpy as np
import traceback

class tester:

    def __init__(self):

        self.path = "test_video/video_test.avi"

        self.frame_properties = {
                "fps":30,
                "height":160,
                "width":160,
                "nb_channels":3,
                "nb_frames":150
            }

        self.topic = "topicTest"
        self.counter_frame_for_test = 0

        self.sld = ShopLiftingDetector.ShopLiftingDetector("ShopLiftingDetector/model_classification.h5")

    def test_callback(self, frame_object):
        prediction = self.sld.predict(frame_object["frames_data"])
        print("the model predicted:", prediction)

        # Create the event video to see the result
        name = "test_video/output/" + str(prediction) + ".avi"
        output = cv2.VideoWriter(name, cv2.VideoWriter_fourcc(*'DIVX'), 30, (160, 160))
        for frame in frame_object["frames_data"]:
            output.write(np.array(frame).astype(np.uint8))
        output.release()

    def prepross(self, frame,frame_properties):
        return cv2.resize(frame, (frame_properties["height"], frame_properties["width"]))

    def prepross2(self, frame, frame_properties):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(gray_frame, (5, 5), 0)

    def event_detector(self, frame, old_frame):
        nb_total = 160 * 160
        delta_frame=cv2.absdiff(old_frame, frame)
        threshold_frame = cv2.threshold(delta_frame, 25, 255, cv2.THRESH_BINARY)[1]
        nb_white = np.sum(threshold_frame == 255)
        ratio = nb_white / nb_total

        event = ratio > 0.005
        if event:
        	print("event detected at frame:",self.counter_frame_for_test)
        self.counter_frame_for_test += 1
        return event

    def test(self):

        self.vok = EventConsumerKafka.EventConsumerKafka(self.test_callback, groupid="test_integration")
        self.vok.start(self.topic)

        time.sleep(2)

        self.vik = EventProducerKafka.EventProducerKafka(self.frame_properties, self.prepross, self.prepross2, self.event_detector)

        self.vik.start(self.path, self.topic)
        time.sleep(120)
        self.vik.stop()
        self.vok.stop()

tester().test()
