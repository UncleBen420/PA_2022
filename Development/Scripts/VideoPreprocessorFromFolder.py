## This script allow user to apply preprocessing on videos
# this script that 3 arguments:
# --path_in: the path where the videos to modify are
# --path_out: the path where the will be saved
# --func: the preprocess methods to apply: opticalFlowSimple, opticalFlowFarneback, pifpaf, pifpaf_opticalflow, rescale, mirror, shift, grayscale
# exemple command line: python3 VideoPreprocessorFromFolder.py --path_in Dataset/ --path_out HomeMadeDataset/ --func rescale

import os
import cv2
import numpy as np
import openpifpaf
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import argparse
import random

class VideoPreprocessorFromFolder:

    def __init__(self):

        self.preprocess_func ={
        "opticalFlowSimple":[self.preprocess_opticalflow_basic, self.transform_opticalflow_basic, False, True],
        "opticalFlowFarneback":[self.preprocess_opticalflow_farneback, self.transform_opticalflow_farneback, True, False],
        "pifpaf":[self.preprocess_pifpaf, self.transform_linear, True, False],
        "pifpaf_opticalflow": [self.preprocess_pifpaf_opticalflow, self.transform_pifpaf_opticalflow, False, True],
        "rescale":[self.preprocess_rescale, self.transform_linear, True, False],
        "mirror":[self.preprocess_mirror, self.transform_linear, True, False],
        "shift":[self.preprocess_shift, self.transform_linear, True, False],
        "grayscale":[self.preprocess_grayscale, self.transform_linear, False, False]
        }

        self.radius = 5
        plasma = cm.get_cmap('plasma', 20)
        self.colors = (plasma(range(20))[:,:3] * 255).astype("uint8")
        self.color = (255, 0, 0)
        self.thickness = -1
        self.squelet_point = [(15,13),(13,11),(16,14),(14,12),(11,12),(5,11),(6,12),
                              (5,6),(5,7),(6,8),(7,9),(8,10),(1,2),(0,1),(0,2),(1,3),(2,4),(3,5),(4,6)]
      #  self.predictor = openpifpaf.Predictor(checkpoint="mobilenetv3small")
        self.predictor = openpifpaf.Predictor(checkpoint="shufflenetv2k16")

        self.height = 160
        self.width  = 160
        self.M = np.float32([[1, 0, -20], [0, 1, -10]])

    def transform_linear(self, frame, old_frame):
        return frame

    def extract_pose(self, predictions, frame):
        for pose in predictions:
            i = 0
            for point in pose.data:
                if (point.all() > 0):
                    im2 = cv2.circle(frame, point[0:2].astype(int), self.radius, self.color, self.thickness)
            for joint in self.squelet_point:
                if pose.data[joint[0]].all() > 0 and pose.data[joint[1]].all() > 0:
                    frame = cv2.line(frame, pose.data[joint[0]][0:2].astype(int), pose.data[joint[1]][0:2].astype(int), self.colors[i].tolist(), 2)
                i += 1
        return frame

    def extract_pose_gray(self, predictions, frame):
        for pose in predictions:
            for point in pose.data:
                if (point.all() > 0):
                    im2 = cv2.circle(frame, point[0:2].astype(int), self.radius, self.color, self.thickness)
            for joint in self.squelet_point:
                if pose.data[joint[0]].all() > 0 and pose.data[joint[1]].all() > 0:
                    frame = cv2.line(frame, pose.data[joint[0]][0:2].astype(int), pose.data[joint[1]][0:2].astype(int), 255, 2)
        return frame

    def preprocess_opticalflow_basic(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.GaussianBlur(frame, (5, 5), 0)
        return frame

    def preprocess_grayscale(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame

    def transform_opticalflow_basic(self, frame, old_frame):
        transform_frame=cv2.absdiff(old_frame, frame)
        return transform_frame

    def preprocess_opticalflow_farneback(self, frame):
        self.mask = np.zeros_like(frame)
        self.mask[:,:,1] = 255
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def transform_opticalflow_farneback(self, frame, old_frame):
        # Calculate dense optical flow by Farneback method
        # https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowfarneback
        flow = cv2.calcOpticalFlowFarneback(old_frame, frame, None, pyr_scale = 0.5, levels = 5, winsize = 15, iterations = 5, poly_n = 3, poly_sigma = 1.1, flags = 0)
        # Compute the magnitude and angle of the 2D vectors
        magnitude, angle = cv2.cartToPolar(flow[:,:,0], flow[:,:,1])
        # Set image hue according to the optical flow direction
        self.mask[:,:,0] = angle * 180 / np.pi / 2
        # Set image value according to the optical flow magnitude (normalized)
        self.mask[:,:,2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        # Convert HSV to RGB (BGR) color representation
        rgb = cv2.cvtColor(self.mask, cv2.COLOR_HSV2BGR)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        return cv2.addWeighted(frame, 1,rgb, 2, 0)

    def preprocess_pifpaf(self, frame):
        blk_image = np.zeros_like(frame)
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        predictions, gt_anns, image_meta = self.predictor.numpy_image(img)
        self.extract_pose(predictions, blk_image)
        return blk_image

    def preprocess_pifpaf_opticalflow(self, frame):
        # for pifpaf model
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # for opticalflow
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.GaussianBlur(frame, (5, 5), 0)

        predictions, gt_anns, image_meta = self.predictor.numpy_image(img)
        self.predictions = predictions
        return frame

    def transform_pifpaf_opticalflow(self, frame, old_frame):
        delta_frame = cv2.absdiff(old_frame, frame)
        self.extract_pose(self.predictions, delta_frame)
        return delta_frame

    def preprocess_rescale(self, frame):
        self.out_height = self.height
        self.out_width  = self.width
        frame = cv2.resize(frame, (self.height, self.width))
        return frame
        
    def preprocess_mirror(self, frame):
        frame = cv2.cv2.flip(frame, 1)
        return frame
        
    def preprocess_shift(self, frame):
    	frame = cv2.warpAffine(frame, self.M, (frame.shape[1], frame.shape[0]))
    	return frame


    def process(self, transform, path_in, path_out):

        config = self.preprocess_func[transform]
        folder = os.listdir(path_in)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')

        for file in folder:

            file_in = os.path.join(path_in, file)
            file_out = os.path.join(path_out, file)
            # setting the input
            input  = cv2.VideoCapture(file_in)

            ret, self.old_frame = input.read()
            if self.old_frame is None:
                print("error video empty")
                input.release()
                break

            if ret:
                # setting the output
                self.out_height = self.old_frame.shape[0]
                self.out_width  = self.old_frame.shape[1]
                fps = input.get(cv2.CAP_PROP_FPS)

                self.old_frame = config[0](self.old_frame)

                output = cv2.VideoWriter(file_out,fourcc,fps,(self.out_width, self.out_height), config[2])

                if config[3]:
                    output.write(np.zeros((self.out_width, self.out_height), dtype = "uint8"))
                else:
                    output.write(self.old_frame)

            while(1):
                ret, frame = input.read()
                if frame is None:
                    print('video ended')
                    break

                if ret:
                    frame = config[0](frame)
                    frame_transformed = config[1](frame.copy(), self.old_frame)
                    output.write(frame_transformed)
                    self.old_frame = frame

            input.release()
            output.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split video into classes by respect of the annotation file')
    parser.add_argument('--path_in', metavar='path', required=True,
                        help='the path to the videos')
    parser.add_argument('--path_out', metavar='path', required=True,
                        help='path where the video will be stored')
    parser.add_argument('--func', required=True,
                        help='preprocessing function applied')

    args = parser.parse_args()

    vpff = VideoPreprocessorFromFolder()

    folder = os.listdir(args.path_in)
    for subfolder in folder:

        folder_in = os.path.join(args.path_in, subfolder)
        folder_out = os.path.join(args.path_out, subfolder + '_' + args.func)

        try:
            os.mkdir(folder_out)
        except OSError as error:
            pass
            #folder olready exist

        vpff.process(args.func, folder_in, folder_out)
