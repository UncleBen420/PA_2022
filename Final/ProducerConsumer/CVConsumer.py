import cv2
import threading

class CVConsumer:

    def connect(self):
        """connect to the broker"""
        pass

    def subscribe(self, topic, callback):
        """subscribe on a topic and listen to upcoming messages"""
        self.listener_topic = topic
        self.listener_callback = callback
        self.shutdown_flag = False
        self.listener = threading.Thread(target=self.listen)
        # to ensure the thread will not be terminated at the wrong time
        self.listener.setDaemon(True)
        self.listener.start()
        
    def listen(self):
        try:
            self.video_input = cv2.VideoCapture(self.listener_topic)
            while not self.shutdown_flag:
                ret, frame = self.video_input.read()
                self.listener_callback(ret, frame)

                if frame is None:
                    # if the video has reach the end
                    self.shutdown_flag = True
                    continue
            self.video_input.release()
        except:
            print(traceback.format_exc())

            self.shutdown_flag = True 
            self.video_input.release()
    
    def unsubscribe(self):
        self.shutdown_flag = True
        self.listener.join()
        self.listener = None
        
    # this method is used when listening on file /!| no need to unsubscribe
    def syncSubscribe(self, topic, callback):
        """subscribe on a topic and listen to upcoming messages"""

        self.listener_topic = topic
        self.listener_callback = callback
        self.video_input = cv2.VideoCapture(self.listener_topic)
        self.video_end = False

        while self.video_input.isOpened() and not self.video_end:
            ret, frame = self.video_input.read()
            if frame is None:
                # if the video has reach the end
                self.video_end = True
                continue
            self.listener_callback(ret, frame)
        self.video_input.release()    
    
    def disconnect(self):
        """disconnect from the broker"""
        pass
