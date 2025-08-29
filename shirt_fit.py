import cv2
import mediapipe as mp
import numpy as np
from mediapipe import solutions
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
mp_selfie_segmentation = mp.solutions.selfie_segmentation

#הסבר שכתבתי בסוף בשביל לחזור על השלבים ולעשות סדר לא צריך לקרוא את ה זה לא באמת רלוונטי זה סתם בשבילי + יש הערות לגבי רוב השורות שכתבתי בהם מה משמעות השורה בשביל לחזור שוב על מה כל דבר עושה
#1. save pics.  2. get height and width from upper body.  3. get the points of the upper body with Mediapipe
#4. save the coordinants of the points of the upper body (x, y).  5. get the top point and bottom point and crop the image
#6. condition - adds the mask three times so it could work, and checks if the pixel is from the body or bg
#7. applies condition on the cropped upper body.  8. make a new picture that is all gray background
#9. create the segmented pic using the condition, if true the pixel is body and if false its background
#10. now we change the size of the shirt pic to fit the screen better for the selection of the rectangle with grab cut, we find the ratio between the measurements to later change the size of the image back
#11. changing the rect size back to the regular measurement using the ratio. 12.making a new mask of all 0 that will ne updated later
#13. using cv2.grabCut on the shirt pic.  14. create a new mask where if grabcut put 2/0 in the mask its background (updated to 0) and the rest is considered the shirt (1)
#15. multiply the image of the shirt with the mask to create a new image of only the shirt
#16. finding the width and height of the upper body using the cordinants we found in the beggining to fit the shirt
#17. finding the center points to place the shirt correctly, but because the shirt is placed according to the side the point we place it by needs to be on the side and not center, just in the same line as the center
#18. find the target width - a little wider than the shoulders, and the target height - a little higher than the center point
#19. resizing the shirt according to the measurements we found to match the body.  20. creating a new mask from the mask of the segmented shirt that is from the numbers 0/255
#21. resizing the mask to fit the body like we did with the actual shirt.  22. making sure the shirt wont be out of bounds
#22. resizing the shirt and mask again according to the new points.  23. taking the area of the body that will be the shirt
#24. making a new mask that leaves the body only where there isnt a shirt.  25. create a shirt area and body area
#26. combine the areas and than do it in the segmented upper body.  27. present the results



image_upper = cv2.imread("Upper_body_front.png") #body pic
image_tshirt = cv2.imread("tshirt2.png") #shirt pic

image_height, image_width, _ = image_upper.shape # body pic measurements
BG_COLOR = (192, 192, 192) #light gray

#static image mode, higher complexity = better results slower work, enables segmantaton
with mp_pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=True, min_detection_confidence=0.5) as pose:
    results = pose.process(cv2.cvtColor(image_upper, cv2.COLOR_BGR2RGB)) #return the points we need


# points of the shoulders, hips and chest
    left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]

#the coordinats for later to dress the person
    leftshoulder_x = int(left_shoulder.x * image_width)
    leftshoulder_y = int(left_shoulder.y * image_height)
    rightshoulder_x = int(right_shoulder.x * image_width)
    rightshoulder_y = int(right_shoulder.y * image_height)

    left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]

    lefthip_x = int(left_hip.x * image_width)
    lefthip_y = int(left_hip.y * image_height)
    righthip_x = int(right_hip.x * image_width)
    righthip_y = int(right_hip.y * image_height)

    top_y = 0
    bottom_y = max(lefthip_y, righthip_y)

    left_x = 0
    right_x = image_width

    cropped_upper_body = image_upper[top_y:bottom_y, left_x:right_x] # returns the picture only of the upper body

    condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1 #multuples the segmentation mask by 3 because it works by 3 channels
    cropped_condition = condition[top_y:bottom_y, left_x:right_x] #applies the condition on the cropped upper body

    bg_image = np.zeros(cropped_upper_body.shape, dtype=np.uint8) # creates a new image of all 0 (bg)
    bg_image[:] = BG_COLOR # does a solid background for the parts that arent the body
    segmented_cropped = np.where(cropped_condition, cropped_upper_body, bg_image) #makes a new pic where if the condition is true its from the body and if false its from the bg_image

    # cv2.imshow("Segmented Upper Body", segmented_cropped)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # changes the size of the pic for the selection of rect
    display_width = 700
    h, w = image_tshirt.shape[:2]
    ratio = display_width / w
    display_height = int(h * ratio)

    resized_for_selection = cv2.resize(image_tshirt, (display_width, display_height))

    #the selection
    rect_resized = cv2.selectROI("Select Shirt", resized_for_selection, fromCenter=False, showCrosshair=True)

    # changing the rect back to original image coordinates after the change for the choosing
    rect = (int(rect_resized[0] / ratio),
            int(rect_resized[1] / ratio),
            int(rect_resized[2] / ratio),
            int(rect_resized[3] / ratio))


    mask = np.zeros(image_tshirt.shape[:2], np.uint8) # creates all zeros mask to use later
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    cv2.grabCut(image_tshirt, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    new_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8') # saves what points are bg and what are the shirt itself
    segmented_tshirt = image_tshirt * new_mask[:, :, np.newaxis] #adds a channel to new mask to mutiple the mask with the image and create a new image of only the shirt

    # cv2.imshow("Segmented T-Shirt", segmented_tshirt)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    shoulder_width = abs(rightshoulder_x - leftshoulder_x) #shoulder width to change the shirt to fit the person
    torso_height = abs(max(lefthip_y, righthip_y) - min(leftshoulder_y, rightshoulder_y)) #the height of the torso so i could adjust the shirt

    tg_width = int(shoulder_width * 2.1) #changing the size of the shirt to fit the person better
    sh, sw, _ = segmented_tshirt.shape # taking the measurements of the segmented shirt
    ratio1 = sh / sw # finding the correct ration between the width and height
    tg_height = int(tg_width * ratio1) #using the ratio to find the target height

    resized_shirt = cv2.resize(segmented_tshirt, (tg_width, tg_height)) # resizing the shirt according to the size of the body

    mask_255 = (new_mask * 255).astype(np.uint8) #changing from 0/1 to 0/255

    resized_mask = cv2.resize(mask_255, (tg_width, tg_height), interpolation=cv2.INTER_NEAREST) #resizing the mask to fit the body

#finding the coorinants to the center to put the shirt on the person
    centerx = (leftshoulder_x + rightshoulder_x) // 2
    centery = (leftshoulder_y + rightshoulder_y) // 2

#בגלל שהחולצה מתחילה משמאל ולא ממרכז צריך למצוא את הנקוד הנכונה שממנה החולצה תתחיל
    shirtx = int(centerx - tg_width // 2)
    shirty = int(centery - 0.22 * tg_height)

    # Make sure the region fits inside the body image
    h_body, w_body, _ = segmented_cropped.shape

    # in case its out of bounds we check and if it is we take the maximun thats still in bounds
    y1 = max(0, shirty)
    y2 = min(shirty + tg_height, h_body)
    x1 = max(0, shirtx)
    x2 = min(shirtx + tg_width, w_body)

    #resizing the shirt again according to what we have
    resized_shirt_cropped = cv2.resize(resized_shirt, (x2 - x1, y2 - y1))
    resized_mask_cropped = cv2.resize(resized_mask, (x2 - x1, y2 - y1))

#הטישטוש לא עבד זה למה שמתי בהערה
    # edges = cv2.Laplacian(resized_mask_cropped, cv2.CV_8U)
    #
    # edges_dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
    #
    # edge_mask_3ch = cv2.merge([edges_dilated] * 3)
    #
    # blurred_shirt = cv2.GaussianBlur(resized_shirt_cropped, (11, 11), 0)
    #
    # resized_shirt_cropped = np.where(edge_mask_3ch > 0, blurred_shirt, resized_shirt_cropped)

    #this determines what is the region on the body the shirt will be in
    roi_in_body = segmented_cropped[y1:y2, x1:x2]

    mask_invert = cv2.bitwise_not(resized_mask_cropped) #a new mask that leaves only the body where there isnt a shirt
    shirt_area = cv2.bitwise_and(resized_shirt_cropped, resized_shirt_cropped, mask=resized_mask_cropped) #makes the shirt
    body_area = cv2.bitwise_and(roi_in_body, roi_in_body, mask=mask_invert) #makes the body

    combined = cv2.add(body_area, shirt_area) #combining the shirt and body
    segmented_cropped[y1:y2, x1:x2] = combined #last step for combining the shirt and body in the wanted area

    cv2.imshow("Final Try-On", segmented_cropped) #presenting results
    cv2.waitKey(0)
    cv2.destroyAllWindows()











































