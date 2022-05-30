import os
import cv2
import numpy as np
import argparse
import pandas as pd

if __name__ == '__main__':
    NB_SECOND = 5
    FPS = 30
    NB_FRAME_PER_VIDEO = NB_SECOND * FPS
    WIDTH = 224
    HEIGHT = 224

    parser = argparse.ArgumentParser(description='Split video into classes by respect of the annotation file')
    parser.add_argument('--path_in', metavar='path', required=True,
                        help='the path to the videos')
    parser.add_argument('--path_out', metavar='path', required=True,
                        help='path where the video will be stored')
    args = parser.parse_args()

    folder = os.listdir(args.path_in)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    for sub_folder in folder:

        print(sub_folder)

        sub_folder_path = os.path.join(args.path_in, sub_folder)
        if os.path.isdir(sub_folder_path):
            print(sub_folder_path)

            sub_folder_element = os.listdir(sub_folder_path)

            # Chaque image
            name =  sub_folder + '1' + '.avi'
            file_out = os.path.join(args.path_out, name)
            output = cv2.VideoWriter(file_out,fourcc,FPS,(WIDTH, HEIGHT),True)

            for file in np.sort(sub_folder_element):

                if os.path.splitext(file)[1] == '.tif':

                    # PREPARATION OF THE FILES
                    file_in = os.path.join(sub_folder_path, file)
                    #file_out = os.path.join(args.path_out, file)

                    input  = cv2.VideoCapture(file_in)

                    ret, frame = input.read()
                    if frame is None:
                        print('error')
                        break
                    frame_transformed = cv2.resize(frame, (WIDTH, HEIGHT))
                    output.write(frame_transformed)

                    input.release()
            output.release()
