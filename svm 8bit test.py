import cv2
import numpy as np
from sklearn import svm

#img=cv2.imread('Mario 8bit.png',1)
img=cv2.imread('kirby_8_bit.png',1)
m=img.shape[0]
n=img.shape[1]
#print(img)

feature=[]; data=[]
for i in range(3):
    data.append([])

for r in range(m):
    for c in range(n):
        feature.append([r,c])
        for i in range(3):
            data[i].append(img[r,c,i])
        
print(len(feature))



clf=svm.SVC()
newImg=np.zeros((2*m,2*n,3))

for i in range(3):
    print("debug1")
    clf.fit(feature,data[i])
    print("debug2")
    for r in range(0,2*m):
        for c in range(0,2*n):
            newImg[r,c,i]=(clf.predict([[r/2,c/2]]))[0]
            
cv2.imwrite('kirby svm.png',newImg)