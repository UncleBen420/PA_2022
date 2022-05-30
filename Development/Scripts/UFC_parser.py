import os
import cv2
import numpy as np
import argparse
import pandas as pd

if __name__ == '__main__':
    NB_SECOND = 5
    FPS = 30
    NB_FRAME_PER_VIDEO = NB_SECOND * FPS
    WIDTH = 299
    HEIGHT = 299

    parser = argparse.ArgumentParser(description='Split video into classes by respect of the annotation file')
    parser.add_argument('--path_in', metavar='path', required=True,
                        help='the path to the videos')
    parser.add_argument('--path_out', metavar='path', required=True,
                        help='path where the video will be stored')
    parser.add_argument('--path_annot', metavar='path', required=True,
                        help='path to the annotation file')
    args = parser.parse_args()
    annotation = pd.read_csv(args.path_annot, delim_whitespace=True, names=["name", "type", "frame_start1", "frame_end1", "frame_start2", "frame_end2"])
    print(annotation)
    folder = os.listdir(args.path_in)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    for file in folder:
        row = annotation[annotation['name'] == file]
        if not row.empty:
            print(file)

            # PREPARATION OF THE FILES
            file_in = os.path.join(args.path_in, file)
            #file_out = os.path.join(args.path_out, file)
            input  = cv2.VideoCapture(file_in)
            nb_frames = int(input.get(cv2.CAP_PROP_FRAME_COUNT))
            # This array of the same size as the number of frames indicate if the frame is suspect or not
            annot_array = np.zeros(nb_frames, dtype="bool")
            annot_array[row["frame_start1"].item():row["frame_end1"].item()] = True

            # if the video contains a second event
            if row["frame_start2"].item() != -1:
                annot_array[row["frame_start2"].item():row["frame_end2"].item()] = True

            for i in range(int(nb_frames / NB_FRAME_PER_VIDEO)):
                if annot_array[i*NB_FRAME_PER_VIDEO:(1+i)*NB_FRAME_PER_VIDEO].any() == True:
                    file_out = os.path.join(args.path_out, "suspect")
                else:
                    file_out = os.path.join(args.path_out, "ras")
                try:
                    os.mkdir(file_out)
                except OSError as error:
                    pass
                    #folder olready exist

                name = file + '_' + str(i) + '.avi'
                file_out = os.path.join(file_out, name)
                output = cv2.VideoWriter(file_out,fourcc,FPS,(WIDTH, HEIGHT),True)

                for _ in range(NB_FRAME_PER_VIDEO):
                    ret, frame = input.read()
                    if frame is None:
                        print('error')
                        break
                    frame_transformed = cv2.resize(frame, (WIDTH, HEIGHT))
                    output.write(frame_transformed)

                output.release()

            input.release()
