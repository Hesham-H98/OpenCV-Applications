import cv2
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

# Global Variables
background = None
accum_weight = 0.5
roi_x1 = 300
roi_x2 = 600
roi_y1 = 30
roi_y2 = 300

# Running average
def accumulative_avg(frame):
    global background

    if background is None:
        background = frame.copy().astype('float')
        return None
    
    cv2.accumulateWeighted(frame, background, accum_weight)

# Segmentation
def segment(frame, threshold=25):

    diff = cv2.absdiff(frame, background.astype('uint8'))

    ret, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    
    segment = max(contours, key=cv2.contourArea)
    return (thresh, segment)

# Finger Counter
def countFingers(thresh, segment):

    convhull = cv2.convexHull(segment)

    top = tuple(convhull[convhull[:,:,1].argmin()][0])
    bottom = tuple(convhull[convhull[:,:,1].argmax()][0])
    right = tuple(convhull[convhull[:,:,0].argmin()][0])
    left = tuple(convhull[convhull[:,:,0].argmax()][0])

    centerX = (right[0] + left[0])//2
    centerY = (top[1] + bottom[1])//2

    distances = euclidean_distances([(centerX,centerY)],[top,bottom,right,left])
    max_dist = distances.max()
    acc_radius = int(max_dist*0.8)
    circumference = 2*np.pi*acc_radius

    circular_roi = np.zeros(thresh.shape[:2], dtype='uint8')
    cv2.circle(circular_roi, (centerX,centerY), acc_radius, 255, 5)
    circular_roi = cv2.bitwise_and(thresh, thresh, mask=circular_roi)

    contours, hierarchy = cv2.findContours(circular_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    count = 0 # Finger count
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)

        out_of_wrist = (y+h) < 1.25*centerY
        limit_points = contour.shape[0] < 0.25*circumference

        if out_of_wrist and limit_points:
            count += 1
    return count

# Main Program
cap = cv2.VideoCapture(0)
num_frames = 0

while True:
    ret, frame = cap.read()

    frame = cv2.flip(frame, 1)

    frame_copy = frame.copy()

    roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_gray = cv2.GaussianBlur(roi_gray, (7, 7), 0)

    if num_frames < 60:
        accumulative_avg(roi_gray)
        if num_frames <= 59:
            cv2.putText(frame_copy, "WAIT! GETTING BACKGROUND AVG.", (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.imshow("Finger Count",frame_copy)
            
    else:
        hand = segment(roi_gray)

        if hand is not None:
            thresholded, hand_segment = hand

            cv2.drawContours(frame_copy, [hand_segment + (roi_x1, roi_y1)], -1, (255, 0, 0),1)

            fingers = countFingers(thresholded, hand_segment)

            cv2.putText(frame_copy, str(fingers), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            cv2.imshow("Thesholded", thresholded)

    cv2.rectangle(frame_copy, (roi_x2, roi_y1), (roi_x1, roi_y2), (0,0,255), 5)

    num_frames += 1

    cv2.imshow("Finger Count", frame_copy)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()