import cv2
import numpy as np
import math


# dimensions of the region of interest:
x, y, w, h = 250, 100, 300, 300


def pre_process_image(frame):

    # Getting region of interest(hand):
    region_of_interest = frame[y:y+h, x:x+w]

    # blurring the image to remove as much noise as possible:
    frame_blur = cv2.GaussianBlur(
        region_of_interest, (7, 7), cv2.BORDER_DEFAULT)

    # changing the color space to YCrCb:
    frame_ycc = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2YCrCb)
    # setting the thresholds for detecting the skin color in the frame and
    # applying a mask which makes skin color white and others black:
    ycc_min = np.array([0, 143, 77])
    ycc_max = np.array([255, 173, 127])
    frame_masked = cv2.inRange(frame_ycc, ycc_min, ycc_max)

    # Morphological operations to clean the image:
    frame_dilate = cv2.dilate(frame_masked, None, 5)

    # Finding the contours in the image:
    contours, hierarchy = cv2.findContours(
        frame_dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return region_of_interest, frame_masked, contours


def hand_gesture_recognition():
    capture = cv2.VideoCapture(0)

    while True:
        success, frame = capture.read()  # getting the video frame
        if not success:
            return

        # flipping the frame and adding the region of interest to the frame
        frame = cv2.flip(frame, 1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 3)

        # Pre-processing the image and getting our region of interest:
        region_of_interest, frame_improved, contours = pre_process_image(frame)

        # Creating a black image to show our convex hull:
        convex_hull_frame = np.zeros(region_of_interest.shape, np.uint8)

        defects = 0

        try:
            # Getting contour with maximum area in out region of interest i.e. hand:
            """
                    iterable - an iterable such as list, tuple, set, dictionary, etc.
                    *iterables (optional) - any number of iterables; can be more than one
                    key (optional) - key function where the iterables are passed and comparison is performed based on its return value
                    default (optional) - default value if the given iterable is empty
            """
            contour_hand = max(
            contours, key=lambda x: cv2.contourArea(x), default=0)

            # Creating a convex hull around the contour:
            convex_hull_hand = cv2.convexHull(contour_hand)

            # Drawing the contour lines for the hand and the convex hull around the hand:
            cv2.drawContours(convex_hull_frame, [
                            contour_hand], 0, (0, 255, 0), 0)
            cv2.drawContours(convex_hull_frame, [
                            convex_hull_hand], 0, (0, 0, 255), 0)

            # Obtaining the convexity defects through the contour lines and the convex hull indices.
            convex_hull_indices = cv2.convexHull(
                contour_hand, returnPoints=False)
            contour_defects = cv2.convexityDefects(
                contour_hand, convex_hull_indices)

            for i in range(contour_defects.shape[0]):
                s, e, f, d = contour_defects[i, 0]
                start = tuple(contour_hand[s][0])
                end = tuple(contour_hand[e][0])
                far = tuple(contour_hand[f][0])

                # finding the angle of the defect using cosine law
                a = math.sqrt((end[0] - start[0]) ** 2 +
                            (end[1] - start[1]) ** 2)
                b = math.sqrt((far[0] - start[0]) ** 2 +
                            (far[1] - start[1]) ** 2)
                c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                angle = (math.acos((b ** 2 + c ** 2 - a ** 2) /
                                (2 * b * c)) * 180) / 3.14

                # we know, angle between 2 fingers is within 90 degrees.
                # so anything greater than that isn;t considered
                if angle <= 100:
                    defects += 1
                    """
                        We iterate over the array rows and draw a line joining the start point and end point,
                        then draw a circle at the farthest point.
                    """
                    cv2.circle(convex_hull_frame, far, 5, (0, 0, 255), -1)

            if defects == 0:
                cv2.putText(frame, "ONE", (50, 50), cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 0, 255), 2)
            elif defects == 1:
                cv2.putText(frame, "TWO", (50, 50), cv2.FONT_HERSHEY_TRIPLEX, 2,(0,0,255), 2)
            elif defects == 2:
                cv2.putText(frame, "THREE", (5, 50), cv2.FONT_HERSHEY_TRIPLEX, 2,(0,0,255), 2)
            elif defects == 3:
                cv2.putText(frame, "FOUR", (50, 50), cv2.FONT_HERSHEY_TRIPLEX, 2,(0,0,255), 2)
            elif defects == 4:
                cv2.putText(frame, "FIVE", (50, 50), cv2.FONT_HERSHEY_TRIPLEX, 2,(0,0,255), 2)
            else:
                pass
        
        except:
            pass

        #displaying result
        cv2.imshow("frame",frame_improved)
        cv2.imshow("convex",convex_hull_frame)
        cv2.imshow("video_frame",frame)

        # return frame, convex_hull_frame

        key_pressed = cv2.waitKey(30) & 0xff
        if key_pressed == 27:
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    hand_gesture_recognition()
