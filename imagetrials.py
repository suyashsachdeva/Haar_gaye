import cv2
import easyocr as ey

#loc = r'C:\Users\suyash\Desktop\cascade (2).xml'
imglink = r'C:\Users\suyash\Desktop\KACHRA\Machine Learing(HURT LOCKER)\Deep Learning\Datasets\positive\p7.jpg'
loc1 = r'C:\Users\suyash\Desktop\KACHRA\laohub\Smile in Pain\zzz...zzz\opencv/haarcascade_frontalface_default.xml'
#dead = r'C:\Users\suyash\Desktop\myhaar.xml'
#new = r'C:\Users\suyash\Desktop\anas_haartrain\haartraining.xml'

# Methods
alg1 = r'C:\Users\suyash\Desktop\KACHRA\Machine Learing(HURT LOCKER)\Deep Learning\DL Cascade\myhaar.xml'
alg2 = r'C:\Users\suyash\Desktop\myhaar.xml'
alg3 = r'C:\Users\suyash\Desktop\cascade (2).xml'
alg4 = r'C:\Users\suyash\Desktop\anas_haartrain\haartraining.xml'

face = cv2.CascadeClassifier(alg1)
# shield = cv2.CascadeClassifier(new)
# deadpool = cv2.CascadeClassifier(dead)

img = cv2.imread(imglink)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face.detectMultiScale(gray, 1.1, 4)

for (x,y,w,h) in faces:
    cv2.rectangle(img, (x,y), (x + w, y + h), (255,0,0), 2)
    # fk = img[y:y+h, x:x+w]
    # shields = shield.detectMultiScale(fk, 1.1, 4)
    # for (x1,y1,w1,h1) in shields:
    #     cv2.rectangle(img, (x + x1, y + y1), (x+x1+w1,y+y1+h1), (0,255,0), 2)
    #     if shields != ():
    #         print(shields)

cv2.imwrite("name.jpg", img)

cv2.waitKey(0)
cv2.destroyAllWindows()

