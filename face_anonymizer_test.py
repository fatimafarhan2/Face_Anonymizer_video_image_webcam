import cv2
import mediapipe as mp
import os
import argparse


args = argparse.ArgumentParser()

# mode can be "image" or "video" or "webcam"
args.add_argument("--mode" , default="webcam")
# FILE PATH for image or video If mode is "image" or "video"
args.add_argument("--filepath" , default=None)
args=args.parse_args()


output_dir="./output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def process_img(img,face_detection):
    img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    out=face_detection.process(img_rgb)
    H,W,_=img.shape


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

            # img= cv2.rectangle(img,(x1,y1),(x1+w,y1+h),(0,255,0),10)

            # blur face
            img[y1:y1+h,x1:x1+w,:]=cv2.blur(img[y1:y1+h,x1:x1+w,:],(30,30))

    return img

# detect face
mp_face_detection=mp.solutions.face_detection

# model_selection=0: short-range model, best for faces within 2 meters from the camera
# model_selection=1: full-range model, best for faces within 10 meters from the camera
with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:

    if args.mode in ["image"]:
        # read image
        img=cv2.imread(args.filepath)


        img=process_img(img,face_detection)

        # save image 
        cv2.imwrite(os.path.join(output_dir,'output.png'),img)
    
    elif args.mode in ["video"]:

                                     
        # read video
        cap=cv2.VideoCapture(args.filepath)

        ret,frame=cap.read()
        output_video=cv2.VideoWriter(os.path.join(output_dir,'output.mp4') ,
cv2.VideoWriter_fourcc(*'mp4v') , 25, (frame.shape[1],int(frame.shape[0]))
        ) 
        while ret:
            frame=process_img(frame,face_detection)
            output_video.write(frame)        
            ret,frame=cap.read()

        cap.release()
        output_video.release()
    elif args.mode in ["webcam"]:
        cap=cv2.VideoCapture(0)

        ret,frame=cap.read()

        while ret:
            frame=process_img(frame,face_detection)

            cv2.imshow("frame",frame)
            
            if cv2.waitKey(1) & 0xFF==ord('q'):
                break
            ret,frame=cap.read()
        

        cap.release()


    



