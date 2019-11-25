import cv2
import numpy as np

def camera(cam):
    capture = cv2.VideoCapture(cam)  #read the video
    flag, frame = capture.read() #read the video in frames
    #gray = fgbg.apply(frame)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)#convert each frame to grayscale.
    blur=cv2.GaussianBlur(gray,(5,5),0)#blur the grayscale image
    ret,th1 = cv2.threshold(blur,100,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)#using threshold remave noise
    ret1,th2 = cv2.threshold(th1,127,255,cv2.THRESH_BINARY_INV)# invert the pixels of the image frame
    im2,contours,hierarchy = cv2.findContours(th2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) #find the contours
    #cv2.drawContours(frame,contours,-1,(0,255,0),3) 
    
    cnt = contours[0]
    hull = cv2.convexHull(cnt)
    x,y,w,h = cv2.boundingRect(cnt)
    #cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),)
    cv2.drawContours(frame, contours, -1, (0,255,0),)

    #print(int(x),int(y))
##        if(y<300 and x<220 and y>20 and x>20):
##                objct_color=frame[int(x+w/2),int(y+w/2)]
##                frame[20:50,20:50]=objct_color
            
    cv2.line(frame,(int(x+w/2),0),(int(x+w/2),240),(0,0,200),1)
    cv2.line(frame,(0,int(y+h/2)),(320,int(y+h/2)),(0,0,200),1)
    #cv2.putText(frame,"cx,xy="+str(x+w/2)+","+str(y+h/2),(10,220),font,1,(0,0,255),1,cv2.LINE_AA)
##        i=1
##        for hullPoints in hull:
##                #cv2.rectangle(frame,tuple(hullPoints[0]),(2,2),(255,255,0),2)
##                cv2.circle(frame,tuple(hullPoints[0]),5, (255,255,0), -1)
##                cv2.putText(frame,str(i),tuple(hullPoints[0]), font, 0.5,(255,255,255),1,cv2.LINE_AA)
##                #print(hullPoints[0])
##                i=i+1
    
    cv2.imshow('ORIGINAL(RGB)'+str(cam),frame) #show video
    #cv2.imshow('GRAY SCALED',gray) #show video
    #cv2.imshow('GRAY SCALED-BLURED',blur) #show video
    #cv2.imshow('THRESOLDED',th1) #show video
    #cv2.imshow('THRESOLD-INVERTED',th2) #show video


##capture.set(3,320.0) #set the size
##capture.set(4,240.0)  #set the size
##capture.set(5,15)  #set the frame rate
x,y=0,0
fgbg = cv2.createBackgroundSubtractorMOG2()
font=cv2.FONT_HERSHEY_SIMPLEX

while cv2.waitKey(1) != 27:
        camera(2)
        camera(0)

cv2.imwrite('ORIGINAL(RGB).jpg',frame)
cv2.imwrite('THRESOLDED.jpg',th1)


#print(contours[0])
#print(hull)
#cv2.destroyAllWindows() 
i=1

print(np.shape(hull))
xval=[]
yval=[]

for hullPoints in hull:
        #cv2.rectangle(frame,tuple(hullPoints[0]),(2,2),(255,255,0),2)
        #cv2.circle(frame,tuple(hullPoints[0]),5, (0,0,255), -1)
        #cv2.putText(frame,str(i),tuple(hullPoints[0]), font, 0.5,(255,255,255),1,cv2.LINE_AA)
        x,y=hullPoints[0]
        #print(x,y)
        xval.append(x)
        yval.append(y)
        i=i+1

print(xval)
print(yval)

xmin=min(xval)
xmax=max(xval)
ymin=min(yval)
ymax=max(yval)

tmpxval=[]
tmpyval=[]


y2=yval[0]
for xv in xval:
        if(abs(xv-xmin)<=2 and y2<yval[xval.index(xv)]):
                y2=yval[xval.index(xv)]
                
x2=xval[0]
for yv in yval:
        if(yv==ymax and x2>xval[yval.index(yv)]):
                x2=xval[yval.index(yv)]
                
#print(xmin,xmax,ymin,ymax)
points=('p1','p2','p3','p4','p5')
p1=(xmin,ymin)
p5=(xmax,ymin)
p2=(xmin,y2)
p3=(x2,ymax)
p4=(xmax,y2)

p6=(x2,ymin+(ymax-y2))

cv2.circle(frame,p1,5, (255,255,0), -1)
cv2.putText(frame,'1',p1, font, 0.5,(255,255,255),1,cv2.LINE_AA)
cv2.circle(frame,p5,5, (255,255,0), -1)
cv2.putText(frame,'5',p5, font, 0.5,(255,255,255),1,cv2.LINE_AA)
cv2.circle(frame,p2,5, (255,255,0), -1)
cv2.putText(frame,'2',p2, font, 0.5,(255,255,255),1,cv2.LINE_AA)
cv2.circle(frame,p3,5, (255,255,0), -1)
cv2.putText(frame,'3',p3, font, 0.5,(255,255,255),1,cv2.LINE_AA)
cv2.circle(frame,p4,5, (255,255,0), -1)
cv2.putText(frame,'4',p4, font, 0.5,(255,255,255),1,cv2.LINE_AA)

cv2.circle(frame,p6,5, (0,0,255), -1)
cv2.putText(frame,'6',p6, font, 0.5,(0,0,255),1,cv2.LINE_AA)

pts = np.array([p1,p2,p3,p4,p5,p6], np.int32)
pts = pts.reshape((-1,1,2))
cv2.polylines(frame,[pts],True,(0,255,255))
cv2.line(frame,tuple(p6),tuple(p3),(0,255,255))
print(p1)
cv2.imshow('NEW',frame)
cv2.imwrite('ORIGINAL(RGB)1.jpg',frame)

