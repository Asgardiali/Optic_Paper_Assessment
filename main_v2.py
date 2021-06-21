import numpy as np
import cv2
import materials

########################################################################
# STARTING PARAMETERS

path = "O2.jpg"
width = 530
height = 660
questions = 15
choices = 5
ans = [0, 0, 2, 3, 4, 1, 1, 2, 2, 4, 1, 1, 2, 3, 4]

########################################################################
# DEFINE IMAGE FOR PROCESSING

img = cv2.imread(path)
img = cv2.resize(img, (width,height))
imgBlank = np.zeros_like(img)
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
imgCanny = cv2.Canny(imgBlur, 10, 50)
imgContours = img.copy()
imgBiggestContours = img.copy()
imgFinal = img.copy()


########################################################################
# FIND ALL CONTOURS
countours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(imgContours, countours, -1, (0, 255, 255), 2)
########################################################################
# FIND THE BIGGEST CONTOURS
rectCon = materials.rectContour(countours)
biggestContour = materials.getCornerPoints(rectCon[0])
gradePoint = materials.getCornerPoints(rectCon[1])

if biggestContour.size != 0 and gradePoint.size != 0:
    cv2.drawContours(imgBiggestContours, biggestContour, -1, (0, 255, 0), 20)
    cv2.drawContours(imgBiggestContours, gradePoint, -1, (255, 0, 0), 20)

    biggestContour = materials.reorder(biggestContour)
    gradePoint = materials.reorder(gradePoint)

    pt1 = np.float32(biggestContour)
    pt2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(pt1, pt2)
    imgWarpColored = cv2.warpPerspective(img, matrix, (width, height))

    imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
    imgTresh = cv2.threshold(imgWarpGray, 180, 250, cv2.THRESH_BINARY_INV)[1]

    boxes = materials.splitBoxes(imgTresh)
    myPixelVal = np.zeros((questions, choices))
    countColm = 0
    countRow = 0

    for image in boxes:
        totalPixels = cv2.countNonZero(image)
        myPixelVal[countRow][countColm] = totalPixels
        countColm += 1
        if (countColm == choices): countRow += 1; countColm = 0

    # Getting marks
    myIndex = []
    for x in range(0, questions):
        arr = myPixelVal[x]
        myIndexVal = np.where(arr == np.amax(arr))
        myIndex.append(myIndexVal[0][0])
    print("My Index:", myIndex)

    # Grade answers
    grading = []
    for x in range(0, questions):
        if ans[x] == myIndex[x]:
            grading.append(1)
        else:
            grading.append(0)
    print(grading)
    score = round((sum(grading) / questions) * 100)
    print("Score = ", score)

    # Displaying Answer
    imgResult = imgWarpColored.copy()
    imgResult = materials.showAnswers(imgResult, myIndex, grading, ans, questions, choices)
    imgRawDrawing = np.zeros_like(imgWarpColored)
    imgRawDrawing = materials.showAnswers(imgRawDrawing, myIndex, grading, ans, questions, choices)
    invmatrix = cv2.getPerspectiveTransform(pt2, pt1)
    imgInvWarp = cv2.warpPerspective(imgRawDrawing, invmatrix, (width, height))

    imgFinal = cv2.addWeighted(imgFinal, 1, imgInvWarp, 1, 0)
    imgFinal = cv2.putText(imgFinal, str(int(score)), (60, 595), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 5)

    cv2.imshow("IMG",imgFinal)
    cv2.waitKey(0)

########################################################################
# SHOW IMAGES

imageArray = ([img, imgGray, imgBlur, imgCanny],
              [imgContours, imgBiggestContours, imgWarpColored, imgTresh],
              [imgResult, imgRawDrawing, imgInvWarp, imgFinal])
labels=[["Original","Gray","Blur","Canny"],
        ["Contours","Biggest Contour","Warp","Threshold"],
        ["Result","Raw Drawing","Inv Warp","Final"]]
imgStacked = materials.stackImages(imageArray, 0.3, labels)

cv2.imshow("Stacked Image", imgStacked)
cv2.waitKey(0)