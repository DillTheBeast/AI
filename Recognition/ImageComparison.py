import cv2
import face_recognition

image1Path = "/Users/dillonmaltese/Documents/git/AI/Recognition/elon1.png"
image2Path = "/Users/dillonmaltese/Documents/git/AI/Recognition/diff.png"

image1 = cv2.imread(image1Path)
image2 = cv2.imread(image2Path)

face1 = face_recognition.face_encodings(image1)
face2 = face_recognition.face_encodings(image2)

face1 = face1[0]
face2 = face2[0]

same = face_recognition.compare_faces([face1], face2)[0]
print(same)

if same:
    distance = face_recognition.face_distance([face1], face2)
    distance = round(distance[0] * 100)
    
    accuracy = 100 - round(distance)
    print("The images are same")
    print(f"Accuracy Level: {accuracy}%")
else:
    print("The images are not same")

