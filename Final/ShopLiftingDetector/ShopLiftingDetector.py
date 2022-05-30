from tensorflow.keras import models
import numpy as np

class ShopLiftingDetector:

    def __init__(self, model="model_classification.h5"):

        self.model =  models.load_model(model)
        self.SEQUENCE_PADDING = 2
        self.SEQUENCE_LENGTH = 15

    def normalise(self, img):
        return (img / 127.5) - 1

    def predict(self, video):
        video = video[0::self.SEQUENCE_PADDING]
        video_frames_count = video.shape[0]
        video = self.normalise(video)

        for count in range(int(video_frames_count / self.SEQUENCE_LENGTH)):

            batch = video[(count*self.SEQUENCE_LENGTH):((count+1)*self.SEQUENCE_LENGTH)]
            pred = self.model.predict(np.array([batch]))

        self.model.reset_states()
        return np.mean(pred)
