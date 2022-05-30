from ProducerConsumer import EventProducerKafka
from ProducerConsumer import EventConsumerKafka
import cv2
import time
import numpy as np

path = "test_video/test_video_kafka.avi"

frame_properties = {
        "fps":10,
        "height":10,
        "width":10,
        "nb_channels":3,
        "nb_frames":5
    }

topic = "topicTest"

def test_callback(frame_object):
    test_array  = np.zeros([5,10,10,3])
    test_array[:][:][:] = [121., 124., 122.]
    assert(np.array_equal(frame_object["frames_data"], test_array))

def prepross(frame,frame_properties):
    return cv2.resize(frame, (frame_properties["height"], frame_properties["width"]))

def prepross2(frame, frame_properties):

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray_frame, (5, 5), 0)

def event_detector(frame, old_frame):
    nb_total = 100
    delta_frame=cv2.absdiff(old_frame, frame)
    threshold_frame = cv2.threshold(delta_frame, 25, 255, cv2.THRESH_BINARY)[1]
    nb_white = np.sum(threshold_frame == 255)

    ratio = nb_white / nb_total

    return ratio > 0.005

vok = EventConsumerKafka.EventConsumerKafka(test_callback, groupid="voktest4")
vok.start(topic)

time.sleep(5)

vik = EventProducerKafka.EventProducerKafka(frame_properties, prepross, prepross2, event_detector)

vik.start(path, topic)
time.sleep(5)
vik.stop()
vok.stop()
