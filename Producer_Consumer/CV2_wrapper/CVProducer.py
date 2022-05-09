import cv2
class CVProducer:
    
    def __init__(self, fourcc):
        self.fourcc = fourcc

    def connect(self):
        """nothing to do"""
        pass

    def init_message(self, topic, frames_descriptor):
        
        name = topic + "/" + str(frames_descriptor["address"]["datetime"]).replace(' ', '_') + ".avi"

        self.out = cv2.VideoWriter(name,
                              self.fourcc,
                              frames_descriptor["frame_properties"]["fps"],
                              (frames_descriptor["frame_properties"]["width"], 
                               frames_descriptor["frame_properties"]["height"]),
                              frames_descriptor["frame_properties"]["channel"])

    def publish(self, topic, frame, callback):
        """publish to others"""
        self.out.write(frame)
        callback(frame)
    
    def flush(self):
        self.out.release()
    
    def disconnect():
        """nothing to do"""
        pass
