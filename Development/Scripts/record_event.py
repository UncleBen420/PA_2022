import cv2
import numpy as np
import time
import datetime
import sys

print('name: ', sys.argv[1])
print('device: ', sys.argv[2])
print('show: ', sys.argv[3])

config = {
    "name":sys.argv[1],
    "dir":"video",
    "record_time":5,
    "threshold": 0.005,
    "device":sys.argv[2],
    "show": int(sys.argv[3])
}

cap = cv2.VideoCapture(int(config['device']))

if not (cap.isOpened()):
    print("Could not open video device")

ret1,old_frame = cap.read()

fshape = old_frame.shape
fheight = fshape[0]
fwidth = fshape[1]
print(fwidth, fheight)
fourcc = cv2.VideoWriter_fourcc(*'XVID')

fps = cap.get(cv2.CAP_PROP_FPS)


old_frame = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
old_frame = cv2.GaussianBlur(old_frame, (5, 5), 0)

nb_total = old_frame.size

record_counter = 0

while(1):

    ret1, new_frame = cap.read()
    gray_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
    delta_frame=cv2.absdiff(old_frame, gray_frame)
    threshold_frame = cv2.threshold(delta_frame, 25, 255, cv2.THRESH_BINARY)[1]

    old_frame = gray_frame
    nb_white = np.sum(threshold_frame == 255)

    ratio = nb_white / nb_total


    stri = '\rratio ' + str(ratio)
    print(stri, end="")

    if config['show'] == 1:
        cv2.imshow('delta2', threshold_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if ratio > config['threshold'] and record_counter == 0:
        record_counter = fps * config['record_time']

        e = datetime.datetime.now()
        dt = e.strftime("%Y-%m-%d_%H.%M.%S")

        event = config['dir'] + '/' + dt + '_' + config['name'] + '.avi'

        out = cv2.VideoWriter(event, fourcc, fps, (fwidth,fheight))
    elif record_counter > 0:
        out.write(new_frame)

        record_counter -= 1
        if record_counter == 0:
            print("event recorded")
            out.release()


cap.release()
cv2.destroyAllWindows()
