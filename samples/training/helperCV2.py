import cv2
import numpy as np
##
## OpenCV (cv2) useful functions
##

## ============================================================================
## READ AND DISPLAY FILES
## ============================================================================
def readImg(file):
    return cv2.imread(file, cv2.IMREAD_UNCHANGED)

def displayImg(img, rectangles = ()):
    img_ = img.copy()

    # Draw rectangles, (xmin, xmax, wmin, ymax)
    for rectangle in rectangles:
        cv2.rectangle(img_, (rectangle[0], rectangle[2]), (rectangle[1], rectangle[3]), (255, 0, 0), 2)

    cv2.imshow('image', img_)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return

def getImgSize(img):
    # returns size as a tuple (width, height, channels)
    result = img.shape
    if len(result) == 3:
        return (result[1], result[0], result[2])
    else:
        return (result[1], result[0], 1)

def saveToFile(img, filename):
    cv2.imwrite(filename,img)

## ============================================================================
## IMAGE TRANSFORMATIONS
## ============================================================================

def reshapeImg(img):
    w, h = getImgSize(img)[0], getImgSize(img)[1] 
    if h > w:
        img = rotateImg(img, 90)
        w, h = getImgSize(img)[0], getImgSize(img)[1]

    if 16*h > 9*w:
        deltay = int(0.5*(h - w*9/16))
        img = traslateImg(img, 0, -1*deltay)
        img = resizeCanvas(img, w, int(w*9/16))
        w, h = getImgSize(img)[0], getImgSize(img)[1]
        img =scaleImg(img, 486/h)
    else:
        deltax = int(0.5*(w - h*16/9))
        img = traslateImg(img, -1*deltax, 0)
        img = resizeCanvas(img, int(16*h/9), h)
        w, h = getImgSize(img)[0], getImgSize(img)[1]
        img =scaleImg(img, 486/h)
    return img

def scaleImg(img, scale_factor):
    return cv2.resize(img, (0,0), fx = scale_factor, fy = scale_factor)

def resizeImg(img, W, H):
    size = getImgSize(img)
    w = size[0]
    h = size[1]
    Fx = W/w
    Fy = H/h
    return cv2.resize(img, (0,0), fx = Fx, fy = Fy)

def resizeCanvas(img, X, Y):
    M = np.float32([[1, 0, 0], [0, 1, 0]])
    return cv2.warpAffine(img, M, (X, Y))

def traslateImg(img, deltaX, deltaY):
    M = np.float32([[1, 0, deltaX], [0, 1, deltaY]])
    size = getImgSize(img)
    size = (size[0],size[1])
    return cv2.warpAffine(img, M, size)

def rotateImg(img, angle_deg):
    # Performs a counter-clockwise rotation

    size = getImgSize(img)
    width = size[0]
    height = size[1]

    angle_rad = angle_deg * np.pi / 180.0
    final_width = int(np.absolute(np.sin(angle_rad)) * height +
                    np.absolute(np.cos(angle_rad) * width))
    final_height = int(np.absolute(np.cos(angle_rad)) * height +
                    np.absolute(np.sin(angle_rad) * width))

    max_dimension = int(np.sqrt(width*width+height*height))+1

    ## 1. Increase the size of the canvas and move to the center
    img = resizeCanvas(img, max_dimension, max_dimension)
    img = traslateImg(img, int((max_dimension - width)/2), int((max_dimension - height)/2))

    ## 2. Rotate around the center
    M = cv2.getRotationMatrix2D(((max_dimension - 1) / 2.0, (max_dimension - 1) / 2.0), angle_deg, 1)
    img = cv2.warpAffine(img, M, (max_dimension, max_dimension))

    ## 3. Resize to the final dimension
    img = traslateImg(img, int((final_width - max_dimension) / 2.0), int((final_height - max_dimension) / 2.0))
    img = resizeCanvas(img, final_width, final_height)

    ##
    return img

def addAlphaChannel(img):
    if getImgSize(img)[2] != 3:
        return img

    b_channel, g_channel, r_channel = cv2.split(img)
    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 50
    return cv2.merge((b_channel, g_channel, r_channel, alpha_channel))

def adjustGamma(image, gamma=1.0):
   invGamma = 1.0 / gamma
   table = np.array([((i / 255.0) ** invGamma) * 255
      for i in np.arange(0, 256)]).astype("uint8")
   return cv2.LUT(image, table)

def placeImg(img_small, img_large, X, Y):
    # places a small image onto a large one at position (X,Y)

    size = getImgSize(img_small)
    width_small  = size[0]
    height_small = size[1]
    size = getImgSize(img_large)
    width_large  = size[0]
    height_large = size[1]

    img_small = resizeCanvas(img_small, width_large, height_large)
    img_small = traslateImg(img_small, X, Y)

    img_small_no_alpha = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img_small_no_alpha, 10, 255, cv2.THRESH_BINARY)
    mask = cv2.bitwise_not(mask)

    img_large = cv2.bitwise_and(img_large, img_large, mask = mask)

    return cv2.add(addAlphaChannel(img_large), img_small)

    #l_img[y_offset:y_offset + s_img.shape[0], x_offset:x_offset + s_img.shape[1]] = s_img
