import cv2
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
mp_pose = mp.solutions.pose
#mp_selfie_segmentation = mp.solutions.selfie_segmentation



def do_grabcut_on_shirt(img_path): ### func to remove the background of the shirt
    img = cv2.imread(img_path)

    assert img is not None, "Problem - file could not be read"

    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    rect = (50, 50, img.shape[1] - 100, img.shape[0] - 100)

    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    new_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    img = img * new_mask[:, :, np.newaxis]

    return img ## returns image of the shirt without the background

    # cv2.imshow("T-Shirt no background", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def get_landmarks(body_img): ## func to get the landmarks on the picture of the body

    base_options = python.BaseOptions(model_asset_path='../pose_landmarker.task')
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=True)
    detector = vision.PoseLandmarker.create_from_options(options)

    img = mp.Image.create_from_file(body_img)

    return detector.detect(img) ## returns landmarks

def debug_landmarks(img_path, landmarks): ## func to make sure the landmarks are correct
    img = cv2.imread(img_path)
    h, w, _ = img.shape
    for idx in [11, 12, 23, 24]:  # shoulders and hips
        x = int(landmarks[idx].x * w)
        y = int(landmarks[idx].y * h)
        cv2.circle(img, (x, y), 6, (0, 255, 0), -1)

    # cv2.imshow("Landmark Debug", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def change_shirt_size(shirt_img, body_img): ##func to change the shirt size according to the body

    results = get_landmarks(body_img)

    if not results.pose_landmarks or len(results.pose_landmarks) == 0:
        raise ValueError("No landmarks detected in the body image.")

    landmarks = results.pose_landmarks[0]

    debug_landmarks(body_img, landmarks)

    body_img_cv = cv2.imread(body_img)
    height, width, _ = body_img_cv.shape

    def to_pixel(landmarks): ## helping func to get the cordinants of the landmarks
        return int(landmarks.x * width), int(landmarks.y * height)

    left_shoulder = to_pixel(landmarks[11])
    right_shoulder = to_pixel(landmarks[12])
    left_hip = to_pixel(landmarks[23])
    right_hip = to_pixel(landmarks[24])

    torso_width = abs(right_shoulder[0] - left_shoulder[0]) * 2.8
    torso_height = abs(left_hip[1] - left_shoulder[1]) * 1.7

    torso_width = max(100, int(torso_width))
    torso_height = max(100, int(torso_height))

    shirt_resized = cv2.resize(shirt_img,(int(torso_width), int(torso_height)), interpolation=cv2.INTER_AREA)

    center_x = int((left_shoulder[0] + right_shoulder[0]) / 2 - torso_width / 2)
    top_y = int(left_shoulder[1]) - int(0.2 * torso_height)
    top_y = max(0, top_y)

    return shirt_resized, (center_x, top_y) ## returns the resized shirt and the cordinants for the center top point


def overlay(shirt_img, body_img, position): ## func to dress the man with the shirt
    x, y = position
    height, width = shirt_img.shape[:2]

    body_h, body_w = body_img.shape[:2]

    if x < 0: shirt_img, x = shirt_img[:, -x:], 0
    if y < 0: shirt_img, y = shirt_img[-y:, :], 0

    if x + width > body_w: shirt_img = shirt_img[:, :body_w - x]
    if y + height > body_h: shirt_img = shirt_img[:body_h - y, :]

    h, w = shirt_img.shape[:2]
    roi = body_img[y:y + h, x:x + w]

    gray = cv2.cvtColor(shirt_img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    roi = body_img[y:y + height, x:x + width]

    if roi.shape[:2] != mask.shape:
        mask = cv2.resize(mask, (roi.shape[1], roi.shape[0]))
        mask_inv = cv2.resize(mask_inv, (roi.shape[1], roi.shape[0]))

    blurred_mask = cv2.GaussianBlur(mask, (15, 15), 0)
    blurred_mask_inv = cv2.bitwise_not(blurred_mask)

    body_bg = cv2.bitwise_and(roi, roi, mask=blurred_mask_inv)
    shirt_fg = cv2.bitwise_and(shirt_img, shirt_img, mask=blurred_mask)

    combined = cv2.add(body_bg, shirt_fg)
    body_img[y:y + height, x:x + width] = combined

    return body_img


def main():

        shirt_img = do_grabcut_on_shirt("WhiteShirt.png")

        body_img = cv2.imread("Upper_body_front.png")

        shirt_resized, position = change_shirt_size(shirt_img, "Upper_body_front.png")

        final_img = overlay(shirt_resized, body_img, position)

        cv2.imshow("Final Dressing", final_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
        main()





