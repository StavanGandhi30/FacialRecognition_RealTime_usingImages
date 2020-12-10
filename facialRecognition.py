import numpy as np
import face_recognition as fr
import cv2

video_capture = cv2.VideoCapture(0)
known_face_encondings = []
known_face_names = []

##You can append any number of images you like.
##For-Loop around more than one image Recommended.
img = fr.load_image_file('stavan.jpeg')
img_encoding = fr.face_encodings(img)[0]
known_face_encondings = [img_encoding]
known_face_names = ['Stavan']   #Name of the person in image

notIdentify = True
while notIdentify: 
    _, frame = video_capture.read()
    rgb_frame = frame[:, :, ::-1]

    face_locations = fr.face_locations(rgb_frame)
    face_encodings = fr.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = fr.compare_faces(known_face_encondings, face_encoding, tolerance=0.47)
        face_distances = fr.face_distance(known_face_encondings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            print(name) #Known Face Name
            unKnown = False
            break

video_capture.release()
cv2.destroyAllWindows()

###<------VARIABLE EXPLAINATION---->
#matches: List of True or False (Matched or Not).
#face_distances: Confidence type. Lower Better.
#best_match_index: Index of Image and Name.
#name: Name of Matched Face.
