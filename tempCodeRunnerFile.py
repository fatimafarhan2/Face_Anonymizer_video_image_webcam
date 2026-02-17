import cv2
import mediapipe as mp
path="./data/testImg.png"

img=cv2.imread(path)

H,W,_=img.shape
# cv2.imshow("image",img)
# cv2.waitKey(0)


# detect face
mp_face_detection=mp.solutions.face_detection

# model_selection=0: short-range model, best for faces within 2 meters from the camera
# model_selection=1: full-range model, best for faces within 10 meters from the camera
with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    out=face_detection.process(img_rgb)

    if out.detections is not None:
        for detection in out.detections:
            location_data=detection.location_data

            bbox=location_data.relative_bounding_box
            #location_data.relative_bounding_box gives us the bounding box of the detected face in relative coordinates (x, y, width, height) where x and y are the coordinates of the top-left corner of the bounding box, and width and height are the dimensions of the bounding box. All values are normalized to the range [0, 1] with respect to the image dimensions.  

            x1,y1,w,h =bbox.xmin, bbox.ymin, bbox.width, bbox.height
            print(x1,y1,w,h)
            # x1,y1,w,h are relative to the image size, so we need to convert them to absolute pixel values

            # convert to absolute pixel values

            # BEFORE: 0.2692332863807678 0.2961292862892151, 0.45934492349624634, 0.306213915348053
            # *********************************************
            # After : 161 266 275 275
            x1=int(x1*W)
            y1=int(y1*H)
            w=int(w*W)
            h=int(h*H)
            print("*"*20)
            # print(x1,y1,w,h)

            img= cv2.rectangle(img,(x1,y1),(x1+w,y1+h),(0,255,0),10)

            # blur face
            img=cv2.blur(img,(10,10))



    cv2.imshow("image",img)
    cv2.waitKey(0)
    



