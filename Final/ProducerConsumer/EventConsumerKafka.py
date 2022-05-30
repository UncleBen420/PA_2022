#!/usr/bin/env python3
from . import KafkaConsumerWrapper
import json
import numpy as np
import traceback
class EventConsumerKafka:

    def __init__(self, callback_analysis, address="localhost", port="29092", groupid="groupbase"):

        self.consumer = KafkaConsumerWrapper.KafkaConsumerWrapper(address, port, groupid)
        self.frames_in_transfert = {}
        self.callback_analysis = callback_analysis

    def callbackSub(self, frame):

        try:
            #/!\ should be replace with somthing like avro
            frame = json.loads(frame.decode('utf-8'))

            # If we receive a frames_descriptor object, we must create a new temporary object
            if frame["type_of"] == "frames_descriptor":
                data = np.zeros([frame["frame_properties"]["nb_frames"],
                                 frame["frame_properties"]["width"],
                                 frame["frame_properties"]["height"],
                                 frame["frame_properties"]["nb_channels"]])

                frame_object = {'frame_properties': frame['frame_properties'],
                                'frames_data': data}

                self.frames_in_transfert[frame["ID"]] = frame_object

            # add the frame data to the temporary object

            elif frame["type_of"] == "frame":
                self.frames_in_transfert[frame["ID"]]["frames_data"][frame["num_frame"]] = frame["data"]

            # when the frame ends
            elif frame["type_of"] == "frames_end":
                self.callback_analysis(self.frames_in_transfert[frame["ID"]])
                self.frames_in_transfert.pop(frame["ID"])
        except:
            print(traceback.format_exc())
            print("bad frame")


    def start(self, topic_in):
        self.consumer.connect()
        self.consumer.subscribe(topic_in, self.callbackSub)

    def stop(self):
        self.consumer.unsubscribe()
