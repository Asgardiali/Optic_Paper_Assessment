import cv2
import numpy as np


## TO STACK ALL THE IMAGES IN ONE WINDOW
def stackImages(imgArray, scale, lables=[]):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        hor_con = np.concatenate(imgArray)
        ver = hor
    if len(lables) != 0:
        eachImgWidth = int(ver.shape[1] / cols)
        eachImgHeight = int(ver.shape[0] / rows)
        # print(eachImgHeight)
        for d in range(0, rows):
            for c in range(0, cols):
                cv2.rectangle(ver, (c * eachImgWidth, eachImgHeight * d),
                              (c * eachImgWidth + len(lables[d][c]) * 13 + 27, 30 + eachImgHeight * d), (255, 255, 255),
                              cv2.FILLED)
                cv2.putText(ver, lables[d][c], (eachImgWidth * c + 10, eachImgHeight * d + 20),
                            cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 255), 2)
    return ver


#
def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))  # REMOVE EXTRA BRACKET
    myPointsNew = np.zeros((4, 1, 2), np.int32)  # NEW MATRIX WITH ARRANGED POINTS
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]  # [0,0]
    myPointsNew[3] = myPoints[np.argmax(add)]  # [w,h]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]  # [w,0]
    myPointsNew[2] = myPoints[np.argmax(diff)]  # [h,0]

    return myPointsNew


# Find the rectangle contour in the optic paper
def rectContour(contours):
    rectCon = []
    max_area = 0
    # Find all image find contours area
    for i in contours:
        area = cv2.contourArea(i)
        # We want to find big rectangle
        if area > 50:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if len(approx) == 4:
                rectCon.append(i)
    rectCon = sorted(rectCon, key=cv2.contourArea, reverse=True)
    # print(len(rectCon))
    return rectCon


# Find rectangle corner point
def getCornerPoints(cont):
    peri = cv2.arcLength(cont, True)  # LENGTH OF CONTOUR
    approx = cv2.approxPolyDP(cont, 0.02 * peri, True)  # APPROXIMATE THE POLY TO GET CORNER POINTS
    return approx


def splitBoxes(img):
    rows = np.vsplit(img, 15)
    boxes = []
    for r in rows:
        cols = np.hsplit(r, 5)
        for box in cols:
            boxes.append(box)
    return boxes


def drawGrid(img, questions, choices):
    secW = int(img.shape[1]/questions)
    secH = int(img.shape[0]/choices)
    for i in range(0, 9):
        pt1 = (0, secH * i)
        pt2 = (img.shape[1], secH * i)
        pt3 = (secW * i, 0)
        pt4 = (secW * i, img.shape[0])
        cv2.line(img, pt1, pt2, (255, 255, 0), 2)
        cv2.line(img, pt3, pt4, (255, 255, 0), 2)

    return img


def showAnswers(img, myIndex, grading, ans, questions=15, choices=5):
    secW = int(img.shape[0]/questions)
    secH = int(img.shape[1]/choices)

    axesLength = (40, 20)
    angle = 0
    startAngle = 0
    endAngle = 360
    color = (255, 0, 0)
    thickness = -1

    for x in range(0,questions):

        myAns = myIndex[x]
        cX = [53, 159, 265, 371, 477]
        cY = [22, 66, 110, 154, 198, 242, 286, 330, 374, 418, 462, 506, 550, 594, 638]

        if grading[x] == 1:
            myColorTrue = (0, 255, 0)
            #cv2.circle(img, (cX[myAns], cY[x]), 20, myColorTrue, cv2.FILLED)
            cv2.ellipse(img, (cX[myAns], cY[x]), axesLength, angle, startAngle, endAngle, myColorTrue, thickness)
        else:
            myColorFalse = (0, 255, 0)
            correctAns = ans[x]
            #cv2.circle(img,(cX[correctAns],cY[x]), 20, myColorFalse, cv2.FILLED)
            cv2.ellipse(img, (cX[correctAns],cY[x]), axesLength, angle, startAngle, endAngle, myColorFalse, thickness)
    return img
