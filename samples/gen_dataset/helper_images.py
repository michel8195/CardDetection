import cv2
from config import *
import random

bord_size = 3  # bord_size alpha=0
alphamask = np.ones((cardH, cardW), dtype=np.uint8) * 255
cv2.rectangle(alphamask, (0, 0), (cardW - 1, cardH - 1), 0, bord_size)
cv2.line(alphamask, (bord_size * 3, 0), (0, bord_size * 3), 0, bord_size)
cv2.line(alphamask, (cardW - bord_size * 3, 0), (cardW, bord_size * 3), 0, bord_size)
cv2.line(alphamask, (0, cardH - bord_size * 3), (bord_size * 3, cardH), 0, bord_size)
cv2.line(alphamask, (cardW - bord_size * 3, cardH), (cardW, cardH - bord_size * 3), 0, bord_size)


def give_me_filename(dirname, suffixes, prefix=""):
    """
        Function that returns a filename or a list of filenames in directory 'dirname'
        that does not exist yet. If 'suffixes' is a list, one filename per suffix in 'suffixes':
        filename = dirname + "/" + prefix + random number + "." + suffix
        Same random number for all the file name
        Ex:
        > give_me_filename("dir","jpg", prefix="prefix")
        'dir/prefix408290659.jpg'
        > give_me_filename("dir",["jpg","xml"])
        ['dir/877739594.jpg', 'dir/877739594.xml']
    """
    if not isinstance(suffixes, list):
        suffixes = [suffixes]

    suffixes = [p if p[0] == '.' else '.' + p for p in suffixes]

    while True:
        bname = "%09d" % random.randint(0, 999999999)
        fnames = []
        for suffix in suffixes:
            fname = os.path.join(dirname, prefix + bname + suffix)
            if not os.path.isfile(fname):
                fnames.append(fname)

        if len(fnames) == len(suffixes): break

    if len(fnames) == 1:
        return fnames[0]
    else:
        return fnames


def varianceOfLaplacian(img):
    """
    Compute the Laplacian of the image and then return the focus
    measure, which is simply the variance of the Laplacian
    Source: A.Rosebrock, https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/
    """
    return cv2.Laplacian(img, cv2.CV_64F).var()


def extract_card(img, output_fn=None, min_focus=120, debug=False):
    """
    """

    imgwarp = None

    # Check the image is not too blurry
    focus = varianceOfLaplacian(img)
    if focus < min_focus:
        if debug: print("Focus too low :", focus)
        return False, None

    # Convert in gray color
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Noise-reducing and edge-preserving filter
    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    # Edge extraction
    edge = cv2.Canny(gray, 30, 200)

    # Find the contours in the edged image
    _, cnts, _ = cv2.findContours(edge.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # We suppose that the contour with largest area corresponds to the contour delimiting the card
    cnt = sorted(cnts, key=cv2.contourArea, reverse=True)[0]

    # We want to check that 'cnt' is the contour of a rectangular shape
    # First, determine 'box', the minimum area bounding rectangle of 'cnt'
    # Then compare area of 'cnt' and area of 'box'
    # Both areas sould be very close
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    areaCnt = cv2.contourArea(cnt)
    areaBox = cv2.contourArea(box)
    valid = areaCnt / areaBox > 0.95

    if valid:
        # We want transform the zone inside the contour into the reference rectangle of dimensions (cardW,cardH)
        ((xr, yr), (wr, hr), thetar) = rect
        # Determine 'Mp' the transformation that transforms 'box' into the reference rectangle
        if wr > hr:
            Mp = cv2.getPerspectiveTransform(np.float32(box), refCard)
        else:
            Mp = cv2.getPerspectiveTransform(np.float32(box), refCardRot)
        # Determine the warped image by applying the transformation to the image
        imgwarp = cv2.warpPerspective(img, Mp, (cardW, cardH))
        # Add alpha layer
        imgwarp = cv2.cvtColor(imgwarp, cv2.COLOR_BGR2BGRA)

        # Shape of 'cnt' is (n,1,2), type=int with n = number of points
        # We reshape into (1,n,2), type=float32, before feeding to perspectiveTransform
        cnta = cnt.reshape(1, -1, 2).astype(np.float32)
        # Apply the transformation 'Mp' to the contour
        cntwarp = cv2.perspectiveTransform(cnta, Mp)
        cntwarp = cntwarp.astype(np.int)

        # We build the alpha channel so that we have transparency on the
        # external border of the card
        # First, initialize alpha channel fully transparent
        alphachannel = np.zeros(imgwarp.shape[:2], dtype=np.uint8)
        # Then fill in the contour to make opaque this zone of the card 
        cv2.drawContours(alphachannel, cntwarp, 0, 255, -1)

        # Apply the alphamask onto the alpha channel to clean it
        alphachannel = cv2.bitwise_and(alphachannel, alphamask)

        # Add the alphachannel to the warped image
        imgwarp[:, :, 3] = alphachannel

        # Save the image to file
        if output_fn is not None:
            cv2.imwrite(output_fn, imgwarp)

    if debug:
        cv2.imshow("Gray", gray)
        cv2.imshow("Canny", edge)
        edge_bgr = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(edge_bgr, [box], 0, (0, 0, 255), 3)
        cv2.drawContours(edge_bgr, [cnt], 0, (0, 255, 0), -1)
        cv2.imshow("Contour with biggest area", edge_bgr)
        if valid:
            cv2.imshow("Alphachannel", alphachannel)
            cv2.imshow("Extracted card", imgwarp)

    return valid, imgwarp


def extract_cards_from_video(video_fn, output_dir=None, keep_ratio=3, min_focus=120, debug=False):
    """
        Extract cards from media file 'video_fn' 
        If 'output_dir' is specified, the cards are saved in 'output_dir'.
        One file per card with a random file name
        Because 2 consecutives frames are probably very similar, we don't use every frame of the video, 
        but only one every 'keep_ratio' frames
        
        Returns list of extracted images
    """
    if not os.path.isfile(video_fn):
        print(f"Video file {video_fn} does not exist !!!")
        return -1, []
    if output_dir is not None and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_fn)
    a = cv2.VideoCapture(video_fn)
    a.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
    # print(a.get(cv2.CAP_PROP_POS_FRAMES))
    # print(a.get(cv2.CAP_PROP_POS_MSEC))

    frame_nb = 0
    imgs_list = []
    while True:
        ret, img = cap.read()
        if not ret: break
        # Work on every 'keep_ratio' frames
        if frame_nb % keep_ratio == 0:
            if output_dir is not None:
                output_fn = give_me_filename(output_dir, "png")
            else:
                output_fn = None
            valid, card_img = extract_card(img, output_fn, min_focus=min_focus, debug=debug)
            if debug:
                k = cv2.waitKey(1)
                if k == 27: break
            if valid:
                imgs_list.append(card_img)
        frame_nb += 1

    if debug:
        cap.release()
        cv2.destroyAllWindows()

    return imgs_list
