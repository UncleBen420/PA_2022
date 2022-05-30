from . import KafkaProducerWrapper
from . import CVConsumer
import json
from datetime import datetime

class EventProducerKafka:

    def __init__(self,
                 frame_properties,
                 frame_preprocess,
                 frame_preprocess_for_detector,
                 event_detector,
                 address="localhost",
                 port="29092",
                 camera_id=0):

        self.producer = KafkaProducerWrapper.KafkaProducerWrapper(address, port)
        self.consumer = CVConsumer.CVConsumer()

        self.frame_properties = frame_properties
        self.frame_preprocess = frame_preprocess
        self.frame_preprocess_for_detector = frame_preprocess_for_detector
        self.event_detector = event_detector

        self.old_frame = None

        self.camera_id = format(camera_id,"08")
        self.monotonic_id = 0
        self.Preproccess_id  = "00000000"
        self.Preprocessor_id = "00000000"

        self.record_counter = 0
        self.num_frame = 0

    def consumerCallback(self, ret, frame):
        if not ret:
            return

        if frame is None:
            if self.record_counter > 0:
                #-------------------------------------------------------------------------------------
                #IF THERE IS AN ERROR AND VIDEO IS RICORDING IT INDICATE THAT THE EVENT ENDS
                self.record_counter = 0
                frames_end = {
                    "ID": ID,
                    "frame_properties":self.frame_properties,
                    "type_of": "frames_end",
                    "datetime": str(datetime.now())
                }
                self.producer.publish(self.topic, json.dumps(frames_end).encode('utf-8'), self.producerCallback)
            return

        #transform frame
        frame = self.frame_preprocess(frame, self.frame_properties)
        frame_event = self.frame_preprocess_for_detector(frame, self.frame_properties)

        if self.old_frame is None:
            self.old_frame = frame_event
            return

        if self.event_detector(frame_event, self.old_frame) and self.record_counter == 0:

            self.record_counter = self.frame_properties["nb_frames"]

            ID = '{0:08b}'.format(2) + self.camera_id + self.Preproccess_id + self.Preprocessor_id
            ID = int(ID, 2)

            #-------------------------------------------------------------------------------------
            # SEND THE BEGINING OF A NEW EVENT
            frames_descriptor = {
                "ID": ID,
                "frame_properties":self.frame_properties,
                "type_of": "frames_descriptor",
                "datetime": str(datetime.now())
            }
            self.producer.publish(self.topic, json.dumps(frames_descriptor).encode('utf-8'), self.producerCallback)

            #-------------------------------------------------------------------------------------
            # SEND THE FIRST FRAME

            frames_descriptor["type_of"] = "frame"
            frames_descriptor["num_frame"] = self.num_frame
            frames_descriptor["data"] = frame.tolist()

            self.producer.publish(self.topic, json.dumps(frames_descriptor).encode('utf-8'), self.producerCallback)
            self.producer.flush()
            self.record_counter -= 1

        elif self.record_counter > 0:
            self.num_frame += 1
            self.record_counter -= 1

            ID = '{0:08b}'.format(2) + self.camera_id + self.Preproccess_id + self.Preprocessor_id
            ID = int(ID, 2)

            #-------------------------------------------------------------------------------------
            #SEND THE FOLLOWING FRAME

            frames_descriptor = {
                "ID": ID,
                "frame_properties":self.frame_properties,
                "type_of": "frame",
                "datetime": str(datetime.now()),
                "num_frame": self.num_frame,
            	 "data": frame.tolist()
            }
            self.producer.publish(self.topic, json.dumps(frames_descriptor).encode('utf-8'), self.producerCallback)
            self.producer.flush()

            if self.record_counter == 0:
                #-------------------------------------------------------------------------------------
                #INDICATE THAT THE EVENT ENDS

                frames_end = {
                    "ID": ID,
                    "frame_properties":self.frame_properties,
                    "type_of": "frames_end",
                    "datetime": str(datetime.now())
                }

                self.producer.publish(self.topic, json.dumps(frames_end).encode('utf-8'), self.producerCallback)
                self.producer.flush()
                self.monotonic_id += 1
                self.num_frame = 0

        self.old_frame = frame_event

    def producerCallback(self, errmsg, frame):
        pass

    def start(self, path, topic):

        self.topic = topic

        self.producer.connect()
        self.consumer.connect()

        self.consumer.subscribe(path, self.consumerCallback)

    def stop(self):
        print("has stop")
        self.consumer.unsubscribe()
        self.producer.disconnect()
        self.consumer.disconnect()
