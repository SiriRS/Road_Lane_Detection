import cv2
import numpy as np

def make_coordinates (image, line_parameters):
    slope, intercept = line_parameters
    y1=image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines):
    left_lines = []
    right_lines = []
    if lines is not None and len(lines) > 0:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            intercept = parameters[1]
            if slope < 0:
                left_lines.append(make_coordinates(image, (slope, intercept)))
            else:
                right_lines.append(make_coordinates(image, (slope, intercept)))
    return [left_lines, right_lines]


def canny(image):
    if image is None:
        return None  # Return None if the input image is empty or None
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)#converting to gray scale
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

def region_of_interest(image):
    height, width = image.shape
    mask = np.zeros_like(image)
    vertices = np.array([
        [(300, height), (1300, height), (550, 200)
    ]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def display_lines(image, average_lines):
    line_image = np.zeros_like(image)
    if average_lines is not None:
        for line in average_lines[0]:  # Left lines
            if line is not None and len(line) == 4:
                x1, y1, x2, y2 = line
                color = (0, 255,0 ) #Green
                cv2.line(line_image, (x1, y1), (x2, y2), color, 5)
        for line in average_lines[1]:  # Right lines
            if line is not None and len(line) == 4:
                x1, y1, x2, y2 = line
                color = (0, 255, 0)  # Green
                cv2.line(line_image, (x1, y1), (x2, y2), color, 5)
    return line_image

capture= cv2.VideoCapture("testing_video.mp4")
while(capture.isOpened()):
    ret, frame = capture.read()
    if not ret or frame is None:
        break
    canny_image=canny(frame)
    cropped_image = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180,100,np.array([]),minLineLength=40,maxLineGap=5)
    average_lines = average_slope_intercept(frame,lines)
    line_image = display_lines(frame,average_lines)
    combo_image =cv2.addWeighted(frame,0.8, line_image ,1,1)
    cv2.imshow("result", combo_image)
    if cv2.waitKey(1) ==ord('q'):
        break
capture.release()
cv2.destroyAllWindows()
