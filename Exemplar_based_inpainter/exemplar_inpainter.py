# uncompyle6 version 3.5.0
# Python bytecode 2.7 (62211)
# Decompiled from: Python 2.7.5 (default, Nov 16 2020, 22:23:17) 
# [GCC 4.8.5 20150623 (Red Hat 4.8.5-44)]
# Embedded file name: Inpainter.py
# Compiled at: 2015-03-21 09:54:03
import sys, os, time, math, cv2, numpy as np

class Inpainter:
    DEFAULT_HALF_PATCH_WIDTH = 3
    MODE_ADDITION = 0
    MODE_MULTIPLICATION = 1
    ERROR_INPUT_MAT_INVALID_TYPE = 0
    ERROR_INPUT_MASK_INVALID_TYPE = 1
    ERROR_MASK_INPUT_SIZE_MISMATCH = 2
    ERROR_HALF_PATCH_WIDTH_ZERO = 3
    CHECK_VALID = 4
    inputImage = None
    mask = updatedMask = None
    result = None
    workImage = None
    sourceRegion = None
    targetRegion = None
    originalSourceRegion = None
    gradientX = None
    gradientY = None
    confidence = None
    data = None
    LAPLACIAN_KERNEL = NORMAL_KERNELX = NORMAL_KERNELY = None
    bestMatchUpperLeft = bestMatchLowerRight = None
    patchHeight = patchWidth = 0
    fillFront = []
    normals = []
    sourcePatchULList = []
    targetPatchSList = []
    targetPatchTList = []
    mode = None
    halfPatchWidth = None
    targetIndex = None

    def __init__(self, inputImage, mask, halfPatchWidth=4, mode=1):
        self.inputImage = np.copy(inputImage)
        self.mask = np.copy(mask)
        self.updatedMask = np.copy(mask)
        self.workImage = np.copy(inputImage)
        self.result = np.ndarray(shape=inputImage.shape, dtype=inputImage.dtype)
        self.mode = mode
        self.halfPatchWidth = halfPatchWidth

    def checkValidInputs(self):
        if not self.inputImage.dtype == np.uint8:
            return self.ERROR_INPUT_MAT_INVALID_TYPE
        if not self.mask.dtype == np.uint8:
            return self.ERROR_INPUT_MASK_INVALID_TYPE
        if not self.mask.shape == self.inputImage.shape[:2]:
            return self.ERROR_MASK_INPUT_SIZE_MISMATCH
        if self.halfPatchWidth == 0:
            return self.ERROR_HALF_PATCH_WIDTH_ZERO
        return self.CHECK_VALID

    def inpaint(self):
        self.initializeMats()
        self.calculateGradients()
        stay = True
        while stay:
            self.computeFillFront()
            self.computeConfidence()
            self.computeData()
            self.computeTarget()
            self.computeBestPatch()
            self.updateMats()
            stay = self.checkEnd()
            cv2.imwrite('../tests/updatedMask.jpg', self.updatedMask)
            cv2.imwrite('../tests/workImage.jpg', self.workImage)

        self.result = np.copy(self.workImage)
        cv2.imshow('Confidence', self.confidence)

    def initializeMats(self):
        _, self.confidence = cv2.threshold(self.mask, 10, 255, cv2.THRESH_BINARY)
        _, self.confidence = cv2.threshold(self.confidence, 2, 1, cv2.THRESH_BINARY_INV)
        self.sourceRegion = np.copy(self.confidence)
        self.sourceRegion = np.uint8(self.sourceRegion)
        self.originalSourceRegion = np.copy(self.sourceRegion)
        self.confidence = np.float32(self.confidence)
        _, self.targetRegion = cv2.threshold(self.mask, 10, 255, cv2.THRESH_BINARY)
        _, self.targetRegion = cv2.threshold(self.targetRegion, 2, 1, cv2.THRESH_BINARY)
        self.targetRegion = np.uint8(self.targetRegion)
        self.data = np.ndarray(shape=self.inputImage.shape[:2], dtype=np.float32)
        self.LAPLACIAN_KERNEL = np.ones((3, 3), dtype=np.float32)
        self.LAPLACIAN_KERNEL[(1, 1)] = -8
        self.NORMAL_KERNELX = np.zeros((3, 3), dtype=np.float32)
        self.NORMAL_KERNELX[(1, 0)] = -1
        self.NORMAL_KERNELX[(1, 2)] = 1
        self.NORMAL_KERNELY = cv2.transpose(self.NORMAL_KERNELX)

    def calculateGradients(self):
        srcGray = cv2.cvtColor(self.workImage, cv2.COLOR_RGB2GRAY)
        self.gradientX = cv2.Scharr(srcGray, cv2.CV_32F, 1, 0)
        self.gradientX = cv2.convertScaleAbs(self.gradientX)
        self.gradientX = np.float32(self.gradientX)
        self.gradientY = cv2.Scharr(srcGray, cv2.CV_32F, 0, 1)
        self.gradientY = cv2.convertScaleAbs(self.gradientY)
        self.gradientY = np.float32(self.gradientY)
        height, width = self.sourceRegion.shape
        for y in range(height):
            for x in range(width):
                if self.sourceRegion[(y, x)] == 0:
                    self.gradientX[(y, x)] = 0
                    self.gradientY[(y, x)] = 0

        self.gradientX /= 255
        self.gradientY /= 255

    def computeFillFront(self):
        boundryMat = cv2.filter2D(self.targetRegion, cv2.CV_32F, self.LAPLACIAN_KERNEL)
        sourceGradientX = cv2.filter2D(self.sourceRegion, cv2.CV_32F, self.NORMAL_KERNELX)
        sourceGradientY = cv2.filter2D(self.sourceRegion, cv2.CV_32F, self.NORMAL_KERNELY)
        del self.fillFront[:]
        del self.normals[:]
        height, width = boundryMat.shape[:2]
        for y in range(height):
            for x in range(width):
                if boundryMat[(y, x)] > 0:
                    self.fillFront.append((x, y))
                    dx = sourceGradientX[(y, x)]
                    dy = sourceGradientY[(y, x)]
                    normalX, normalY = dy, -dx
                    tempF = math.sqrt(pow(normalX, 2) + pow(normalY, 2))
                    if not tempF == 0:
                        normalX /= tempF
                        normalY /= tempF
                    self.normals.append((normalX, normalY))

    def getPatch(self, point):
        centerX, centerY = point
        height, width = self.workImage.shape[:2]
        minX = max(centerX - self.halfPatchWidth, 0)
        maxX = min(centerX + self.halfPatchWidth, width - 1)
        minY = max(centerY - self.halfPatchWidth, 0)
        maxY = min(centerY + self.halfPatchWidth, height - 1)
        upperLeft = (minX, minY)
        lowerRight = (maxX, maxY)
        return (
         upperLeft, lowerRight)

    def computeConfidence(self):
        for p in self.fillFront:
            pX, pY = p
            (aX, aY), (bX, bY) = self.getPatch(p)
            total = 0
            for y in range(aY, bY + 1):
                for x in range(aX, bX + 1):
                    if self.targetRegion[(y, x)] == 0:
                        total += self.confidence[(y, x)]

            self.confidence[(pY, pX)] = total / ((bX - aX + 1) * (bY - aY + 1))

    def computeData(self):
        for i in range(len(self.fillFront)):
            x, y = self.fillFront[i]
            currentNormalX, currentNormalY = self.normals[i]
            self.data[(y, x)] = math.fabs(self.gradientX[(y, x)] * currentNormalX + self.gradientY[(y, x)] * currentNormalY) + 0.001

    def computeTarget(self):
        self.targetIndex = 0
        maxPriority, priority = (0, 0)
        omega, alpha, beta = (0.7, 0.2, 0.8)
        for i in range(len(self.fillFront)):
            x, y = self.fillFront[i]
            Rcp = (1 - omega) * self.confidence[(y, x)] + omega
            priority = alpha * Rcp + beta * self.data[(y, x)]
            if priority > maxPriority:
                maxPriority = priority
                self.targetIndex = i

    def computeBestPatch(self):
        minError = bestPatchVariance = 9999999999999999
        currentPoint = self.fillFront[self.targetIndex]
        (aX, aY), (bX, bY) = self.getPatch(currentPoint)
        pHeight, pWidth = bY - aY + 1, bX - aX + 1
        height, width = self.workImage.shape[:2]
        workImage = self.workImage.tolist()
        if pHeight != self.patchHeight or pWidth != self.patchWidth:
            print 'patch size changed.'
            self.patchHeight, self.patchWidth = pHeight, pWidth
            area = pHeight * pWidth
            SUM_KERNEL = np.ones((pHeight, pWidth), dtype=np.uint8)
            convolvedMat = cv2.filter2D(self.originalSourceRegion, cv2.CV_8U, SUM_KERNEL, anchor=(0,
                                                                                                  0))
            self.sourcePatchULList = []
            for y in range(height - pHeight):
                for x in range(width - pWidth):
                    if convolvedMat[(y, x)] == area:
                        self.sourcePatchULList.append((y, x))

        countedNum = 0
        self.targetPatchSList = []
        self.targetPatchTList = []
        for i in range(pHeight):
            for j in range(pWidth):
                if self.sourceRegion[(aY + i, aX + j)] == 1:
                    countedNum += 1
                    self.targetPatchSList.append((i, j))
                else:
                    self.targetPatchTList.append((i, j))

        for y, x in self.sourcePatchULList:
            patchError = 0
            meanR = meanG = meanB = 0
            skipPatch = False
            for i, j in self.targetPatchSList:
                sourcePixel = workImage[(y + i)][(x + j)]
                targetPixel = workImage[(aY + i)][(aX + j)]
                for c in range(3):
                    difference = float(sourcePixel[c]) - float(targetPixel[c])
                    patchError += math.pow(difference, 2)

                meanR += sourcePixel[0]
                meanG += sourcePixel[1]
                meanB += sourcePixel[2]

            countedNum = float(countedNum)
            patchError /= countedNum
            meanR /= countedNum
            meanG /= countedNum
            meanB /= countedNum
            alpha, beta = (0.9, 0.5)
            if alpha * patchError <= minError:
                patchVariance = 0
                for i, j in self.targetPatchTList:
                    sourcePixel = workImage[(y + i)][(x + j)]
                    difference = sourcePixel[0] - meanR
                    patchVariance += math.pow(difference, 2)
                    difference = sourcePixel[1] - meanG
                    patchVariance += math.pow(difference, 2)
                    difference = sourcePixel[2] - meanB
                    patchVariance += math.pow(difference, 2)

                if patchError < alpha * minError or patchVariance < beta * bestPatchVariance:
                    bestPatchVariance = patchVariance
                    minError = patchError
                    self.bestMatchUpperLeft = (x, y)
                    self.bestMatchLowerRight = (x + pWidth - 1, y + pHeight - 1)

    def updateMats(self):
        targetPoint = self.fillFront[self.targetIndex]
        tX, tY = targetPoint
        (aX, aY), (bX, bY) = self.getPatch(targetPoint)
        bulX, bulY = self.bestMatchUpperLeft
        pHeight, pWidth = bY - aY + 1, bX - aX + 1
        for i, j in self.targetPatchTList:
            self.workImage[(aY + i, aX + j)] = self.workImage[(bulY + i, bulX + j)]
            self.gradientX[(aY + i, aX + j)] = self.gradientX[(bulY + i, bulX + j)]
            self.gradientY[(aY + i, aX + j)] = self.gradientY[(bulY + i, bulX + j)]
            self.confidence[(aY + i, aX + j)] = self.confidence[(tY, tX)]
            self.sourceRegion[(aY + i, aX + j)] = 1
            self.targetRegion[(aY + i, aX + j)] = 0
            self.updatedMask[(aY + i, aX + j)] = 0

    def checkEnd(self):
        height, width = self.sourceRegion.shape[:2]
        for y in range(height):
            for x in range(width):
                if self.sourceRegion[(y, x)] == 0:
                    return True

        return False