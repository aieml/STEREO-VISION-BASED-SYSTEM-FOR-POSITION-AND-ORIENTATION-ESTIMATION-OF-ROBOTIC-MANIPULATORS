import numpy as np
import cv2
import tkinter as tk
from PIL import Image, ImageTk
import math as m
import numpy as np
from statistics import mean
import time as t
import matplotlib.pyplot as plt
from matplotlib import style
import serial
import struct
########################################################################################################################

#arduino=serial.Serial('COM12',9600)

key=' '
objectD=' '

fxR=823.98175049
fyR=818.34320068
cxR=297.0671601 
cyR=215.63753475

fxL=802.47924805
fyL=802.65380859
cxL=331.67814387
cyL=237.88057137

ps=0.0028
focal_length=fyR*ps
baseLine=110

mx,my,mz,mtheta=0,0,0,0
bx,by,bz,btheta=0,0,0,0

predict=0

font=cv2.FONT_HERSHEY_SIMPLEX

train=False
test=False
enter=False
tCount=0
pts=[]

CubeP2xm=1
CubeP2ym=1
CubeP2zm=1

CubeP3xm=1
CubeP3ym=1
CubeP3zm=1


CubeP2xc=0
CubeP2yc=0
CubeP2zc=0

CubeP3xc=0
CubeP3yc=0
CubeP3zc=0


CylinderP1xm=1
CylinderP1ym=1
CylinderP1zm=1

CylinderP2xm=1
CylinderP2ym=1
CylinderP2zm=1

CylinderP3xm=1
CylinderP3ym=1
CylinderP3zm=1

CylinderP4xm=1
CylinderP4ym=1
CylinderP4zm=1


CylinderP1xc=0
CylinderP1yc=0
CylinderP1zc=0

CylinderP2xc=0
CylinderP2yc=0
CylinderP2zc=0

CylinderP3xc=0
CylinderP3yc=0
CylinderP3zc=0

CylinderP4xc=0
CylinderP4yc=0
CylinderP4zc=0


SphereP1xm=1
SphereP1ym=1
SphereP1zm=1

SphereP2xm=1
SphereP2ym=1
SphereP2zm=1

SphereP1xc=0
SphereP1yc=0
SphereP1zc=0

SphereP2xc=0
SphereP2yc=0
SphereP2zc=0

########################################################################################################################

camL=0
camR=2

window = tk.Tk()
window.wm_title("STEREO VISION")
window.config(background="white")

imageFrameL = tk.Frame(window, width=600, height=500)
imageFrameL.grid(row=0, column=0,columnspan=4, padx=10, pady=2)

imageFrameR = tk.Frame(window, width=600, height=500)
imageFrameR.grid(row=0, column=4, columnspan=4,padx=10, pady=2)

lmainL = tk.Label(imageFrameL)
lmainL.grid(row=0, column=0,columnspan=4)

lmainR = tk.Label(imageFrameR)
lmainR.grid(row=0, column=4,columnspan=4)

cameraL = cv2.VideoCapture(camL)
cameraR = cv2.VideoCapture(camR)

labelStatus=tk.Label(window,text='STATUS: Cameras Ready | Press START',bg='gray20',fg='white',font='Helvetica 18 bold',width=88,height=1)
labelStatus.grid(row=2,columnspan=8,padx=5,pady=5)

labelStatus2=tk.Label(window,text='Predicted Word Coordinates (ROBOT):  ------------------------- ',bg='gray20',fg='white',font='Helvetica 18 bold',width=88,height=1)
labelStatus2.grid(row=3,columnspan=8,padx=5,pady=5)

btnPreview=tk.Button(window,text='START',bg='navy',fg='white',font='Helvetica 12 bold',width=10,height=1,command=lambda:btnprs('1'))
btnCapture=tk.Button(window,text='PREDICT',bg='navy',fg='white',font='Helvetica 12 bold',width=10,height=1,command=lambda:btnprs('2'))
btnNext=tk.Button(window,text='NEXT',bg='navy',fg='white',font='Helvetica 12 bold',width=10,height=1,command=lambda:btnprs('3'))
btnTrain=tk.Button(window,text='TRAIN',bg='navy',fg='white',font='Helvetica 12 bold',width=10,height=1,command=lambda:btnprs('4'))
btnExit=tk.Button(window,text='EXIT',bg='red',fg='white',font='Helvetica 12 bold',width=12,height=1,command=window.destroy)
btnEnter=tk.Button(window,text='ENTER',bg='gray20',fg='white',font='Helvetica 12 bold',width=7,height=1,command=lambda:btnprs('5'))
btnTest=tk.Button(window,text='TEST',bg='navy',fg='white',font='Helvetica 12 bold',width=10,height=1,command=lambda:btnprs('6'))

btnPreview.grid(row=1,column=0,padx=6,pady=8)
btnCapture.grid(row=1,column=1,padx=6,pady=8)
btnNext.grid(row=1,column=2,padx=6,pady=8)
btnTrain.grid(row=1,column=3,padx=6,pady=8)
btnExit.grid(row=1,column=7,padx=8,pady=8)
btnEnter.grid(row=1,column=6,padx=8,pady=8)
btnTest.grid(row=1,column=4,padx=6,pady=8)

entry=tk.Entry(window,width=20,bg='gray90',font='Helvetica 17')
entry.grid(row=1,column=5)

def btnprs(k):
    
    global key
    global train
    global test
    global pts
    
    key=k

    if(key=='3'):
        key='x'
        labelStatus.config(text='STATUS: Cameras Ready | Press START',font='Helvetica 18 bold',width=88,height=1)
        labelStatus2.config(text='Predicted Word Coordinates (ROBOT):  ------------------------- ')
        main()

    elif(key=='4'):
        key='x'
        train=True
        labelStatus.config(text='STATUS: Cameras Ready | TRAINING MODE, Press START',bg='red')
        labelStatus2.config(text='Predicted Word Coordinates (ROBOT):  ------------------------- ')
        main()

    elif(key=='6'):
        key='x'
        test=True
        labelStatus.config(text='STATUS: Cameras Ready | TESTING MODE, Press START',bg='green')
        labelStatus2.config(text='Predicted Word Coordinates (ROBOT):  ------------------------- ')
        main()

    elif(key=='5' and train==True):

        ptsR=entry.get()
        entry.delete(0, 'end')
        train_data(pts,ptsR,objectD)

    elif(key=='5' and test==True):

        ptsR=entry.get()
        entry.delete(0, 'end')
        test_model(pts,ptsR,objectD)

    elif(key=='5'):

        msg=entry.get()
        entry.delete(0, 'end')

        if(msg=='send'):

            gripper(pts,objectD)

        elif(msg=='rel'):

            msg=msg+'\n'
            arduino.write(msg.encode('UTF-8'))

def centerCoords(p1,p2,objectD):

    if(objectD=='CUBE'):

        thetaY=m.atan((p2[2]-p1[2])/(p2[0]-p1[0]))
        thetaY=abs(thetaY)
        
        x1=p1[0]
        y1=p1[2]
        x2=p2[0]
        y2=p2[2]

        length=pow((pow((x1-x2),2)+pow((y1-y2),2)),0.5)

        x4=x1+length*np.sin(thetaY)
        y4=y1+length*np.cos(thetaY)

        x3=x2+length*np.sin(thetaY)
        y3=y2+length*np.cos(thetaY)

        thetaY=-157.764-round(thetaY*180/m.pi)

        xc=x1+((x3-x1)/2)
        yc=y2+((y4-y2)/2)

        xc=114.08-round(xc)
        yc=687.2+round(yc)

        labelStatus2.config(text='Predicted Word Coordinates (ROBOT): (Xc,Yc,Zc)='+str(yc)+','+str(xc)+','+str(47)+'  theta(Z)='+str(thetaY))

    elif(objectD=='CYLINDER'):

        x1=p1[0]
        y1=p1[2]
        x2=p2[0]
        y2=p2[2]

        xc=x1+25
        yc=y1

        xc=114.08-round(xc)
        yc=687.2+round(yc)

        labelStatus2.config(text='Predicted Word Coordinates (ROBOT): (Xc,Yc,Zc)='+str(yc)+','+str(xc)+','+str(47))
        

    elif(objectD=='SPHERE'):

        x1=p1[0]
        y1=p1[2]
        x2=p2[0]
        y2=p2[2]

        xc=x1+25
        yc=y1

        xc=114.08-round(xc)
        yc=687.2+round(yc)


        labelStatus2.config(text='Predicted Word Coordinates (ROBOT): (Xc,Yc,Zc)='+str(yc)+','+str(xc)+','+str(47))

def gripper(pts,objectD):
    
    length=0

    if(objectD=='CUBE'):

        xp2=pts[0][0]
        zp2=pts[0][2]

        xp3=pts[1][0]
        zp3=pts[1][2]

        dx=abs(xp3-xp2)
        dz=abs(zp3-zp2)
        
        length=pow((pow(dx,2)+pow(dz,2)),0.5)

        length=round(length)
        length=int(length)
        length=str(length)

        msg=str(length)+'\n'
        arduino.write(msg.encode('UTF-8'))
        labelStatus.config(text='STATUS: Predicted Gripper Jaw Length='+str(length)+'mm | Sent to Controller')

    elif(objectD=='CYLINDER'):

        xp2=pts[0][0]
        xp3=pts[1][0]
        dx=abs(xp3-xp2)

        length=dx

        length=round(length)
        length=int(length)
        length=str(length)

        msg=str(length)+'\n'
        arduino.write(msg.encode('UTF-8'))
        labelStatus.config(text='STATUS: Predicted Gripper Jaw Length='+str(length)+'mm | Sent to Controller')

    elif(objectD=='SPHERE'):

        xp2=pts[0][0]
        xp3=pts[1][0]
        dx=abs(xp3-xp2)

        length=dx

        length=round(length)
        length=int(length)
        length=str(length)

        msg=str(length)+'\n'
        arduino.write(msg.encode('UTF-8'))
        labelStatus.config(text='STATUS: Predicted Gripper Jaw Length='+str(length)+'mm | Sent to Controller')        
        
    #arduino.close()
        
def test_model(ptsP,ptsR,objectD):

    ptsR=eval('['+ptsR+']')

    def accuracy(predicted,real):

        acc=(abs(real-predicted)*100)/abs(real)
        acc=100-acc
        return round(acc,1)

    if(objectD=='CUBE'):

        p2xp=ptsP[0][0]
        p2yp=ptsP[0][1]
        p2zp=ptsP[0][2]

        p3xp=ptsP[1][0]
        p3yp=ptsP[1][1]
        p3zp=ptsP[1][2]

        thetaYp=m.atan((p3zp-p2zp)/(p3xp-p2xp))
        thetaYp=round(thetaYp*180/m.pi)      


        p2xr=ptsR[0][0]
        p2yr=ptsR[0][1]
        p2zr=ptsR[0][2]

        p3xr=ptsR[1][0]
        p3yr=ptsR[1][1]
        p3zr=ptsR[1][2]

        thetaYr=m.atan((p3zr-p2zr)/(p3xr-p2xr))
        thetaYr=round(thetaYr*180/m.pi)

        #labelStatus.config(text='Predicted Coords: P2='+str([p2xp,p2yp,p2zp])+' P3='+str([p3xp,p3yp,p3zp])+' thetaY='+str(thetaYp)+' | Actual Coords: P2='+str([p2xr,p2yr,p2zr])+' P3='+str([p3xr,p3yr,p3zr])+' thetaY='+str(thetaYr),font='Helvetica 12 bold',width=120,height=1)
        labelStatus.config(text='Accuracy%: P2='+str(accuracy(p2zr,p2zp))+' P3='+str(accuracy(p3zr,p3zp))+' thetaY='+str(accuracy(thetaYr,thetaYp)),font='Helvetica 12 bold',width=120,height=1)
        
    elif(objectD=='CYLINDER'):

        p1xp=ptsP[0][0]
        p1yp=ptsP[0][1]
        p1zp=ptsP[0][2]

        p2xp=ptsP[1][0]
        p2yp=ptsP[1][1]
        p2zp=ptsP[1][2]

        p3xp=ptsP[2][0]
        p3yp=ptsP[2][1]
        p3zp=ptsP[2][2]

        p4xp=ptsP[3][0]
        p4yp=ptsP[3][1]
        p4zp=ptsP[3][2]


        p1xr=ptsR[0][0]
        p1yr=ptsR[0][1]
        p1zr=ptsR[0][2]

        p2xr=ptsR[1][0]
        p2yr=ptsR[1][1]
        p2zr=ptsR[1][2]

        p3xr=ptsR[2][0]
        p3yr=ptsR[2][1]
        p3zr=ptsR[2][2]

        p4xr=ptsR[3][0]
        p4yr=ptsR[3][1]
        p4zr=ptsR[3][2]

        labelStatus.config(text='Accuracy%: P1='+str(accuracy(p1zr,p1zp))+' P2='+str(accuracy(p2zr,p2zp))+' P3='+str(accuracy(p3zr,p3zp))+' P4='+str(accuracy(p4zr,p4zp)),font='Helvetica 12 bold',width=120,height=1)

    elif(objectD=='SPHERE'):

        p1xp=ptsP[0][0]
        p1yp=ptsP[0][1]
        p1zp=ptsP[0][2]

        p2xp=ptsP[1][0]
        p2yp=ptsP[1][1]
        p2zp=ptsP[1][2]


        p1xr=ptsR[0][0]
        p1yr=ptsR[0][1]
        p1zr=ptsR[0][2]

        p2xr=ptsR[1][0]
        p2yr=ptsR[1][1]
        p2zr=ptsR[1][2]

        labelStatus.config(text='Accuracy%: P1='+str(accuracy(p1zr,p1zp))+' P2='+str(accuracy(p2zr,p2zp)),font='Helvetica 12 bold',width=120,height=1)


def train_data(ptsP,ptsR,objectD):

    global key

    trainDataCube=open("train_data/trainDataCube.txt", 'a')
    trainDataCylinder=open("train_data/trainDataCylinder.txt", 'a')
    trainDataSphere=open("train_data/trainDataSphere.txt", 'a')

    
    if(ptsR=='end'):

        key='x'
        train=False
        labelStatus.config(text='STATUS: Cameras Ready | Press START',bg='gray20')
        main()

    else:

        if(objectD=='CUBE'):

            p2p=ptsP[0]
            p3p=ptsP[1]
            
            trainDataCube.write('[')
            trainDataCube.write(str(p2p))
            trainDataCube.write(',')
            trainDataCube.write(str(p3p))
            trainDataCube.write(',')
            trainDataCube.write(str(ptsR))  
            trainDataCube.write(']')
            trainDataCube.write('\n')

            trainDataCube.close()

        elif(objectD=='CYLINDER'):

            p1p=ptsP[0]
            p2p=ptsP[1]
            p3p=ptsP[2]
            p4p=ptsP[3]
            
            
            trainDataCylinder.write('[')
            trainDataCylinder.write(str(p1p))
            trainDataCylinder.write(',')
            trainDataCylinder.write(str(p2p))
            trainDataCylinder.write(',')
            trainDataCylinder.write(str(p3p))
            trainDataCylinder.write(',')
            trainDataCylinder.write(str(p4p))
            trainDataCylinder.write(',')
            trainDataCylinder.write(str(ptsR))  
            trainDataCylinder.write(']')
            trainDataCylinder.write('\n')

            trainDataCylinder.close()


        elif(objectD=='SPHERE'):

            p1p=ptsP[0]
            p2p=ptsP[1]
            
            trainDataSphere.write('[')
            trainDataSphere.write(str(p1p))
            trainDataSphere.write(',')
            trainDataSphere.write(str(p2p))
            trainDataSphere.write(',')
            trainDataSphere.write(str(ptsR))  
            trainDataSphere.write(']')
            trainDataSphere.write('\n')

            trainDataSphere.close()


def train_model():

    Cubep2xpVals=[]
    Cubep2ypVals=[]
    Cubep2zpVals=[]

    Cubep3xpVals=[]
    Cubep3ypVals=[]
    Cubep3zpVals=[]

    Cubep2xrVals=[]
    Cubep2yrVals=[]
    Cubep2zrVals=[]

    Cubep3xrVals=[]
    Cubep3yrVals=[]
    Cubep3zrVals=[]
    

    Cylinderp1xpVals=[]
    Cylinderp1ypVals=[]
    Cylinderp1zpVals=[]

    Cylinderp2xpVals=[]
    Cylinderp2ypVals=[]
    Cylinderp2zpVals=[]

    Cylinderp3xpVals=[]
    Cylinderp3ypVals=[]
    Cylinderp3zpVals=[]

    Cylinderp4xpVals=[]
    Cylinderp4ypVals=[]
    Cylinderp4zpVals=[]


    Cylinderp1xrVals=[]
    Cylinderp1yrVals=[]
    Cylinderp1zrVals=[]

    Cylinderp2xrVals=[]
    Cylinderp2yrVals=[]
    Cylinderp2zrVals=[]

    Cylinderp3xrVals=[]
    Cylinderp3yrVals=[]
    Cylinderp3zrVals=[]

    Cylinderp4xrVals=[]
    Cylinderp4yrVals=[]
    Cylinderp4zrVals=[]

    Spherep1xpVals=[]
    Spherep1ypVals=[]
    Spherep1zpVals=[]

    Spherep2xpVals=[]
    Spherep2ypVals=[]
    Spherep2zpVals=[]

    Spherep1xrVals=[]
    Spherep1yrVals=[]
    Spherep1zrVals=[]

    Spherep2xrVals=[]
    Spherep2yrVals=[]
    Spherep2zrVals=[]
    

    global CubeP2xm
    global CubeP2ym
    global CubeP2zm

    global CubeP3xm
    global CubeP3ym
    global CubeP3zm


    global CubeP2xc
    global CubeP2yc
    global CubeP2zc
    
    global CubeP3xc
    global CubeP3yc
    global CubeP3zc


    global CylinderP1xm
    global CylinderP1ym
    global CylinderP1zm

    global CylinderP2xm
    global CylinderP2ym
    global CylinderP2zm

    global CylinderP3xm
    global CylinderP3ym
    global CylinderP3zm

    global CylinderP4xm
    global CylinderP4ym
    global CylinderP4zm


    global CylinderP1xc
    global CylinderP1yc
    global CylinderP1zc

    global CylinderP2xc
    global CylinderP2yc
    global CylinderP2zc

    global CylinderP3xc
    global CylinderP3yc
    global CylinderP3zc

    global CylinderP4xc
    global CylinderP4yc
    global CylinderP4zc


    global SphereP1xm
    global SphereP1ym
    global SphereP1zm

    global SphereP2xm
    global SphereP2ym
    global SphereP2zm

    global SphereP1xc
    global SphereP1yc
    global SphereP1zc

    global SphereP2xc
    global SphereP2yc
    global SphereP2zc

    regressionData=open("train_data/regressionData.txt", 'w')

    def linear_regression(xs,ys,graphName):
        
        xs=np.array(xs,dtype=np.float64)
        ys=np.array(ys,dtype=np.float64)

        m = (((mean(xs)*mean(ys)) - mean(xs*ys)) /((mean(xs)*mean(xs)) - mean(xs*xs)))
        b = mean(ys) - m*mean(xs)

        regression_line = [(m*x)+b for x in xs]

        style.use('ggplot')
        plt.title('Regression for '+graphName)
        plt.xlabel('Predicted '+graphName+' (mm)')
        plt.ylabel('Actual '+graphName+' (mm)')
        plt.scatter(xs,ys,color='#003F72',label='data')
        plt.plot(xs, regression_line, label='regression line')
        plt.legend(loc=4)
        plt.savefig('train_data/plots/Linear Regression for '+graphName+'.png')
        plt.close() 

        regressionData.write(graphName)
        regressionData.write(',')
        regressionData.write(str(m))
        regressionData.write(',')
        regressionData.write(str(b))
        regressionData.write('\n')

        return m, b

    trainDataCube=open("train_data/trainDataCube.txt", 'r')
    trainDataCylinder=open("train_data/trainDataCylinder.txt", 'r')
    trainDataSphere=open("train_data/trainDataSphere.txt", 'r')

    dataCube=trainDataCube.readlines()

    for data in dataCube:

        data=eval(data)
        
        p2xp=data[0][0]
        p2yp=data[0][1]
        p2zp=data[0][2]

        p3xp=data[1][0]
        p3yp=data[1][1]
        p3zp=data[1][2]


        p2xr=data[2][0]
        p2yr=data[2][1]
        p2zr=data[2][2]

        p3xr=data[3][0]
        p3yr=data[3][1]
        p3zr=data[3][2]


        Cubep2xpVals.append(p2xp)
        Cubep2ypVals.append(p2yp)
        Cubep2zpVals.append(p2zp)

        Cubep3xpVals.append(p3xp)
        Cubep3ypVals.append(p3yp)
        Cubep3zpVals.append(p3zp)

        Cubep2xrVals.append(p2xr)
        Cubep2yrVals.append(p2yr)
        Cubep2zrVals.append(p2zr)

        Cubep3xrVals.append(p3xr)
        Cubep3yrVals.append(p3yr)
        Cubep3zrVals.append(p3zr)

    dataCylinder=trainDataCylinder.readlines()

    for data in dataCylinder:

        data=eval(data)

        p1xp=data[0][0]
        p1yp=data[0][1]
        p1zp=data[0][2]

        p2xp=data[1][0]
        p2yp=data[1][1]
        p2zp=data[1][2]

        p3xp=data[2][0]
        p3yp=data[2][1]
        p3zp=data[2][2]

        p4xp=data[3][0]
        p4yp=data[3][1]
        p4zp=data[3][2]



        p1xr=data[4][0]
        p1yr=data[4][1]
        p1zr=data[4][2]

        p2xr=data[5][0]
        p2yr=data[5][1]
        p2zr=data[5][2]

        p3xr=data[6][0]
        p3yr=data[6][1]
        p3zr=data[6][2]

        p4xr=data[7][0]
        p4yr=data[7][1]
        p4zr=data[7][2]


        Cylinderp1xpVals.append(p1xp)
        Cylinderp1ypVals.append(p1yp)
        Cylinderp1zpVals.append(p1zp)

        Cylinderp2xpVals.append(p2xp)
        Cylinderp2ypVals.append(p2yp)
        Cylinderp2zpVals.append(p2zp)

        Cylinderp3xpVals.append(p3xp)
        Cylinderp3ypVals.append(p3yp)
        Cylinderp3zpVals.append(p3zp)

        Cylinderp4xpVals.append(p4xp)
        Cylinderp4ypVals.append(p4yp)
        Cylinderp4zpVals.append(p4zp)


        Cylinderp1xrVals.append(p1xr)
        Cylinderp1yrVals.append(p1yr)
        Cylinderp1zrVals.append(p1zr)

        Cylinderp2xrVals.append(p2xr)
        Cylinderp2yrVals.append(p2yr)
        Cylinderp2zrVals.append(p2zr)

        Cylinderp3xrVals.append(p3xr)
        Cylinderp3yrVals.append(p3yr)
        Cylinderp3zrVals.append(p3zr)

        Cylinderp4xrVals.append(p4xr)
        Cylinderp4yrVals.append(p4yr)
        Cylinderp4zrVals.append(p4zr)

    dataSphere=trainDataSphere.readlines()

    for data in dataSphere:

        data=eval(data)

        p1xp=data[0][0]
        p1yp=data[0][1]
        p1zp=data[0][2]

        p2xp=data[1][0]
        p2yp=data[1][1]
        p2zp=data[1][2]


        p1xr=data[2][0]
        p1yr=data[2][1]
        p1zr=data[2][2]

        p2xr=data[3][0]
        p2yr=data[3][1]
        p2zr=data[3][2]


        Spherep1xpVals.append(p1xp)
        Spherep1ypVals.append(p1yp)
        Spherep1zpVals.append(p1zp)

        Spherep2xpVals.append(p2xp)
        Spherep2ypVals.append(p2yp)
        Spherep2zpVals.append(p2zp)


        Spherep1xrVals.append(p1xr)
        Spherep1yrVals.append(p1yr)
        Spherep1zrVals.append(p1zr)

        Spherep2xrVals.append(p2xr)
        Spherep2yrVals.append(p2yr)
        Spherep2zrVals.append(p2zr)

    (CubeP2xm,CubeP2xc)=linear_regression(Cubep2xpVals,Cubep2xrVals,'P2(x coordinate)- CUBES')
    (CubeP2ym,CubeP2yc)=linear_regression(Cubep2ypVals,Cubep2yrVals,'P2(y coordinate)- CUBES')
    (CubeP2zm,CubeP2zc)=linear_regression(Cubep2zpVals,Cubep2zrVals,'P2(z coordinate)- CUBES')
    
    (CubeP3xm,CubeP3xc)=linear_regression(Cubep3xpVals,Cubep3xrVals,'P3(x coordinate)- CUBES')
    (CubeP3ym,CubeP3yc)=linear_regression(Cubep3ypVals,Cubep3yrVals,'P3(y coordinate)- CUBES')
    (CubeP3zm,CubeP3zc)=linear_regression(Cubep3zpVals,Cubep3zrVals,'P3(z coordinate)- CUBES')


    (CylinerP1xm,CylinderP1xc)=linear_regression(Cylinderp1xpVals,Cylinderp1xrVals,'P1(x coordinate)- CYLINDERS')
    (CylinerP1ym,CylinderP1yc)=linear_regression(Cylinderp1ypVals,Cylinderp1yrVals,'P1(y coordinate)- CYLINDERS')
    (CylinerP1zm,CylinderP1zc)=linear_regression(Cylinderp1zpVals,Cylinderp1zrVals,'P1(z coordinate)- CYLINDERS')
    
    (CylinerP2xm,CylinderP2xc)=linear_regression(Cylinderp2xpVals,Cylinderp2xrVals,'P2(x coordinate)- CYLINDERS')
    (CylinerP2ym,CylinderP2yc)=linear_regression(Cylinderp2ypVals,Cylinderp2yrVals,'P2(y coordinate)- CYLINDERS')
    (CylinerP2zm,CylinderP2zc)=linear_regression(Cylinderp2zpVals,Cylinderp2zrVals,'P2(z coordinate)- CYLINDERS')

    #print(CylinerP2zm,CylinderP2zc)

    (CylinerP3xm,CylinderP3xc)=linear_regression(Cylinderp3xpVals,Cylinderp3xrVals,'P3(x coordinate)- CYLINDERS')
    (CylinerP3ym,CylinderP3yc)=linear_regression(Cylinderp3ypVals,Cylinderp3yrVals,'P3(y coordinate)- CYLINDERS')
    (CylinerP3zm,CylinderP3zc)=linear_regression(Cylinderp3zpVals,Cylinderp3zrVals,'P3(z coordinate)- CYLINDERS')

    (CylinerP4xm,CylinderP4xc)=linear_regression(Cylinderp4xpVals,Cylinderp4xrVals,'P4(x coordinate)- CYLINDERS')
    (CylinerP4ym,CylinderP4yc)=linear_regression(Cylinderp4ypVals,Cylinderp4yrVals,'P4(y coordinate)- CYLINDERS')
    (CylinerP4zm,CylinderP4zc)=linear_regression(Cylinderp4zpVals,Cylinderp4zrVals,'P4(z coordinate)- CYLINDERS')


    (SphereP1xm,SphereP1xc)=linear_regression(Spherep1xpVals,Spherep1xrVals,'P1(x coordinate)- SPHERES')
    (SphereP1ym,SphereP1yc)=linear_regression(Spherep1ypVals,Spherep1yrVals,'P1(y coordinate)- SPHERES')
    (SphereP1zm,SphereP1zc)=linear_regression(Spherep1zpVals,Spherep1zrVals,'P1(z coordinate)- SPHERES')

    (SphereP2xm,SphereP2xc)=linear_regression(Spherep2xpVals,Spherep2xrVals,'P2(x coordinate)- SPHERES')
    (SphereP2ym,SphereP2yc)=linear_regression(Spherep2ypVals,Spherep2yrVals,'P2(y coordinate)- SPHERES')
    (SphereP2zm,SphereP2zc)=linear_regression(Spherep2zpVals,Spherep2zrVals,'P2(z coordinate)- SPHERES')

    trainDataCube.close()
    trainDataCylinder.close()
    trainDataSphere.close()
    regressionData.close()
      
               
def main():

    global objectD
    global train
    global test
    global tCount
    global pts
    
    def p6_algorithmL(hull,frame,th2,train):

        global tCount
        global font
        global dispL
        
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


        #print(xval)
        #print(yval)

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
        p1=[xmin,ymin]
        p5=[xmax,ymin]
        p2=[xmin,y2]
        p3=[x2,ymax]
        p4=[xmax,y2]

        p6=[x2,ymin+(ymax-y2)]
        #x6=xval[yval.index(ymin)]
        #p6=[x6,ymin]

        cv2.circle(frame,tuple(p1),5, (255,255,0), -1)
        cv2.putText(frame,'1',tuple(p1), font, 0.5,(255,255,255),1,cv2.LINE_AA)
        cv2.circle(frame,tuple(p5),5, (255,255,0), -1)
        cv2.putText(frame,'5',tuple(p5), font, 0.5,(255,255,255),1,cv2.LINE_AA)
        cv2.circle(frame,tuple(p2),5, (255,0,0), -1)
        cv2.putText(frame,'2',tuple(p2), font, 0.5,(255,255,255),1,cv2.LINE_AA)
        cv2.circle(frame,tuple(p3),5, (0,0,255), -1)
        cv2.putText(frame,'3',tuple(p3), font, 0.5,(255,255,255),1,cv2.LINE_AA)
        cv2.circle(frame,tuple(p4),5, (255,255,0), -1)
        cv2.putText(frame,'4',tuple(p4), font, 0.5,(255,255,255),1,cv2.LINE_AA)

        cv2.circle(frame,tuple(p6),5, (0,255,255), -1)
        cv2.putText(frame,'6',tuple(p6), font, 0.5,(0,0,255),1,cv2.LINE_AA)

        pts = np.array([p1,p2,p3,p4,p5,p6], np.int32)
        pts = pts.reshape((-1,1,2))
        cv2.polylines(frame,[pts],True,(0,255,255))
        cv2.line(frame,tuple(p6),tuple(p3),(0,255,255))

        if(train==True):
            
            cv2.imwrite('Samples/Cube/ORIGINAL(RGB)-L '+str(tCount)+'.jpg',frame)
            cv2.imwrite('Samples/Cube/THRESOLDED-L '+str(tCount)+'.jpg',th2)

        return p2,p3
        
    def p6_algorithmR(hull,frame,th2,train):

        global tCount
        global font
        global dispR
        
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


        #print(xval)
        #print(yval)

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
                        
        points=('p1','p2','p3','p4','p5')
        p1=[xmin,ymin]
        p5=[xmax,ymin]
        p2=[xmin,y2]
        p3=[x2,ymax]
        p4=[xmax,y2]

        p6=[x2,ymin+(ymax-y2)]
        #x6=xval[yval.index(ymin)]
        #p6=[x6,ymin]

        cv2.circle(frame,tuple(p1),5, (255,255,0), -1)
        cv2.putText(frame,'1',tuple(p1), font, 0.5,(255,255,255),1,cv2.LINE_AA)
        cv2.circle(frame,tuple(p5),5, (255,255,0), -1)
        cv2.putText(frame,'5',tuple(p5), font, 0.5,(255,255,255),1,cv2.LINE_AA)
        cv2.circle(frame,tuple(p2),5, (255,0,0), -1)
        cv2.putText(frame,'2',tuple(p2), font, 0.5,(255,255,255),1,cv2.LINE_AA)
        cv2.circle(frame,tuple(p3),5, (0,0,255), -1)
        cv2.putText(frame,'3',tuple(p3), font, 0.5,(255,255,255),1,cv2.LINE_AA)
        cv2.circle(frame,tuple(p4),5, (255,255,0), -1)
        cv2.putText(frame,'4',tuple(p4), font, 0.5,(255,255,255),1,cv2.LINE_AA)

        cv2.circle(frame,tuple(p6),5, (0,255,255), -1)
        cv2.putText(frame,'6',tuple(p6), font, 0.5,(0,0,255),1,cv2.LINE_AA)

        pts = np.array([p1,p2,p3,p4,p5,p6], np.int32)
        pts = pts.reshape((-1,1,2))
        cv2.polylines(frame,[pts],True,(0,255,255))
        cv2.line(frame,tuple(p6),tuple(p3),(0,255,255))

        if(train==True):
        
            cv2.imwrite('Samples/Cube/ORIGINAL(RGB)-R '+str(tCount)+'.jpg',frame)
            cv2.imwrite('Samples/Cube/THRESOLDED-R '+str(tCount)+'.jpg',th2)

        return p2,p3
        
    def p4_algorithmL(hull,frame,th2,train):

        global tCount
        global font
        global dispL
        
        x,y,w,h=cv2.boundingRect(hull)

        p1=(x,y)
        p2=(x,y+h)
        p3=(x+w,y+h)
        p4=(x+w,y)

        cv2.circle(frame,tuple(p1),5, (255,255,0), -1)
        cv2.putText(frame,'1',tuple(p1), font, 0.5,(255,255,255),1,cv2.LINE_AA)
        cv2.circle(frame,tuple(p2),5, (255,255,0), -1)
        cv2.putText(frame,'2',tuple(p2), font, 0.5,(255,255,255),1,cv2.LINE_AA)
        cv2.circle(frame,tuple(p3),5, (0,0,255), -1)
        cv2.putText(frame,'3',tuple(p3), font, 0.5,(255,255,255),1,cv2.LINE_AA)
        cv2.circle(frame,tuple(p4),5,(0,0,255), -1)
        cv2.putText(frame,'4',tuple(p4), font, 0.5,(255,255,255),1,cv2.LINE_AA)

        pts = np.array([p1,p2,p3,p4], np.int32)
        pts = pts.reshape((-1,1,2))
        cv2.polylines(frame,[pts],True,(0,255,255))

        if(train==True):
                  
            cv2.imwrite('Samples/Cylinder/ORIGINAL(RGB)-L '+str(tCount)+'.jpg',frame)
            cv2.imwrite('Samples/Cylinder/THRESOLDED-L '+str(tCount)+'.jpg',th2)
        #print(p1,p4)

        return p1,p2,p3,p4

    def p4_algorithmR(hull,frame,th2,train):

        global tCount
        global font
        global dispL
        
        x,y,w,h=cv2.boundingRect(hull)

        p1=(x,y)
        p2=(x,y+h)
        p3=(x+w,y+h)
        p4=(x+w,y)

        cv2.circle(frame,tuple(p1),5, (255,255,0), -1)
        cv2.putText(frame,'1',tuple(p1), font, 0.5,(255,255,255),1,cv2.LINE_AA)
        cv2.circle(frame,tuple(p2),5, (255,255,0), -1)
        cv2.putText(frame,'2',tuple(p2), font, 0.5,(255,255,255),1,cv2.LINE_AA)
        cv2.circle(frame,tuple(p3),5, (0,0,255), -1)
        cv2.putText(frame,'3',tuple(p3), font, 0.5,(255,255,255),1,cv2.LINE_AA)
        cv2.circle(frame,tuple(p4),5, (0,0,255), -1)
        cv2.putText(frame,'4',tuple(p4), font, 0.5,(255,255,255),1,cv2.LINE_AA)

        pts = np.array([p1,p2,p3,p4], np.int32)
        pts = pts.reshape((-1,1,2))
        cv2.polylines(frame,[pts],True,(0,255,255))

        if(train==True):
            
            cv2.imwrite('Samples/Cylinder/ORIGINAL(RGB)-R '+str(tCount)+'.jpg',frame)
            cv2.imwrite('Samples/Cylinder/THRESOLDED-R '+str(tCount)+'.jpg',th2)

        #print(p1,p4)
        return p1,p2,p3,p4

    def p2_algorithmL(hull,frame,th2,train):

        global tCount
        global font
        global dispL
        
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


        #print(xval)
        #print(yval)

        xmin=min(xval)
        xmax=max(xval)
        ymin=min(yval)
        ymax=max(yval)

        tmpxval=[]
        tmpyval=[]


        yxmin=yval[xval.index(xmin)]
        yxmax=yval[xval.index(xmax)]
        
        p1=[xmin,yxmin]
        p2=[xmax,yxmax]
        center=[int(xmin+((xmax-xmin)/2)),int(yxmax)]
        radius=int((xmax-xmin)/2)
        
        cv2.circle(frame,tuple(p1),5, (255,255,0), -1)
        cv2.putText(frame,'1',tuple(p1), font, 0.5,(255,255,255),1,cv2.LINE_AA)
        cv2.circle(frame,tuple(p2),5, (255,255,0), -1)
        cv2.putText(frame,'2',tuple(p2), font, 0.5,(255,255,255),1,cv2.LINE_AA)

        cv2.circle(frame,tuple(center),radius,(0,255,255),1)

        if(train==True):
            
            cv2.imwrite('Samples/Sphere/ORIGINAL(RGB)-L '+str(tCount)+'.jpg',frame)
            cv2.imwrite('Samples/Sphere/THRESOLDED-L '+str(tCount)+'.jpg',th2)

        return p1,p2    
        
    def p2_algorithmR(hull,frame,th2,train):

        global tCount
        global font
        global dispL
        
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


        #print(xval)
        #print(yval)

        xmin=min(xval)
        xmax=max(xval)
        ymin=min(yval)
        ymax=max(yval)

        tmpxval=[]
        tmpyval=[]


        yxmin=yval[xval.index(xmin)]
        yxmax=yval[xval.index(xmax)]
        
        p1=[xmin,yxmin]
        p2=[xmax,yxmax]
        center=[int(xmin+((xmax-xmin)/2)),int(yxmax)]
        radius=int((xmax-xmin)/2)

        cv2.circle(frame,tuple(p1),5, (255,255,0), -1)
        cv2.putText(frame,'1',tuple(p1), font, 0.5,(255,255,255),1,cv2.LINE_AA)
        cv2.circle(frame,tuple(p2),5, (255,255,0), -1)
        cv2.putText(frame,'2',tuple(p2), font, 0.5,(255,255,255),1,cv2.LINE_AA)
        cv2.circle(frame,tuple(center),radius,(0,255,255),1)

        if(train==True):
        
            cv2.imwrite('Samples/Sphere/ORIGINAL(RGB)-L '+str(tCount)+'.jpg',frame)
            cv2.imwrite('Samples/Sphere/THRESOLDED-L '+str(tCount)+'.jpg',th2)

        return p1,p2
    

    def calZ(ptsL,ptsR,objct):
        
        global ps
        global focal_length
        global baseLine
        global cxL
        global cyL
        global cxR
        global cyR
        global fxL
        global fyL
        global fxR
        global fyR

        disp=[]
        depth=[]
        realPtsL=[]
        thetaY=0

        for i in range(len(ptsL)):

            ds=abs(ptsL[i][0]-ptsR[i][0])
            disp.append(ds)

            dp=int((4*float(baseLine))/(float(ds)*ps))
            depth.append(dp)

            rx=int((ptsL[i][0]-cxL)*dp/fxL)
            ry=int((ptsL[i][1]-cyL)*dp/fyL)

            realPtsL.append([rx,ry,dp])        
        
        return realPtsL

            
######################################################################################### end of inner functions

    ret, frameL = cameraL.read()
    ret, frameR = cameraR.read()
    
    if(key=='1'):

        w,h=frameL.shape[:2]
        Left_Stereo_Map= cv2.initUndistortRectifyMap(mtxL,distL,None,OmtxL,(w,h),5)   
        frame_niceL= cv2.remap(frameL,Left_Stereo_Map[0],Left_Stereo_Map[1], cv2.INTER_LINEAR,0)  
        x,y,w,h=roiL
        frame_niceL=frame_niceL[y:y+h,x:x+w]

        w,h=frameR.shape[:2]
        Right_Stereo_Map= cv2.initUndistortRectifyMap(mtxR,distR,None,OmtxR,(w,h),5)
        frame_niceR= cv2.remap(frameR,Right_Stereo_Map[0],Right_Stereo_Map[1], cv2.INTER_LINEAR,0)
        x,y,w,h=roiR
        frame_niceR=frame_niceR[y:y+h,x:x+w]


        grayL=cv2.cvtColor(frame_niceL,cv2.COLOR_BGR2GRAY)
        grayL = cv2.bilateralFilter(grayL,9,75,75)
        ret,thL = cv2.threshold(grayL,100,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        ret,thL = cv2.threshold(thL,127,255,cv2.THRESH_BINARY_INV)

        grayR=cv2.cvtColor(frame_niceR,cv2.COLOR_BGR2GRAY)
        grayR = cv2.bilateralFilter(grayR,9,75,75)
        ret,thR = cv2.threshold(grayR,100,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        ret,thR = cv2.threshold(thR,127,255,cv2.THRESH_BINARY_INV)

        canny_image = cv2.Canny(thR,250,255)
        canny_image = cv2.convertScaleAbs(canny_image)
        kernel = np.ones((3,3), np.uint8)
        dilated_image = cv2.dilate(canny_image,kernel,iterations=1)

        ret,contours, h = cv2.findContours(dilated_image, 1, 2)
        contours= sorted(contours, key = cv2.contourArea, reverse = True)[:1]
        pt = (180, 3 * frame_niceR.shape[0] // 4)

        for cnt in contours:

            if(cv2.contourArea(cnt)>1000):
            
                approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
                #print(len(approx))

                if len(approx) ==6 :

                    objectD='CUBE'

                elif len(approx) == 7 or len(approx) == 8:

                    objectD='CUBE'

                elif (len(approx) == 5):

                    objectD='CUBE'
                    
                elif len(approx) > 9:


                    objectD='SPHERE'
            else:

                objectD='NOTHING'
            #print(len(approx))   
            cv2.drawContours(frame_niceR,[cnt],-1,(0,255,0),2)
            cv2.putText(frame_niceR,objectD, pt ,font, 0.5,[0,255,0], 2)
            labelStatus.config(text='STATUS: '+objectD+' Detected | Press PREDICT to Approximate the Center of Volume')
        
        imgL = Image.fromarray(thL)
        imgR = Image.fromarray(thR)
        
        imgtkL = ImageTk.PhotoImage(image=imgL)
        imgtkR = ImageTk.PhotoImage(image=imgR)
         
        lmainL.imgtk = imgtkL
        lmainR.imgtk = imgtkR
        
        lmainL.configure(image=imgtkL)
        lmainR.configure(image=imgtkR)
        
        window.after(100, main)
        
    elif(key=='2'):
        
        w,h=frameL.shape[:2]
        Left_Stereo_Map= cv2.initUndistortRectifyMap(mtxL,distL,None,OmtxL,(w,h),5)   # cv2.CV_16SC2 this format enables us the programme to work faster
        frame_niceL= cv2.remap(frameL,Left_Stereo_Map[0],Left_Stereo_Map[1], cv2.INTER_LINEAR,0)  # Rectify the image using the kalibration parameters founds during the initialisation
        x,y,w,h=roiL
        frame_niceL=frame_niceL[y:y+h,x:x+w]

        w,h=frameR.shape[:2]
        Right_Stereo_Map= cv2.initUndistortRectifyMap(mtxR,distR,None,OmtxR,(w,h),5)
        frame_niceR= cv2.remap(frameR,Right_Stereo_Map[0],Right_Stereo_Map[1], cv2.INTER_LINEAR,0)
        x,y,w,h=roiR
        frame_niceR=frame_niceR[y:y+h,x:x+w]


        grayL=cv2.cvtColor(frame_niceL,cv2.COLOR_BGR2GRAY)
        blurL=cv2.GaussianBlur(grayL,(5,5),0)
        ret,thL = cv2.threshold(blurL,100,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        ret,thL = cv2.threshold(thL,127,255,cv2.THRESH_BINARY_INV)
        im,contoursL,hierarchy = cv2.findContours(thL,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        grayR=cv2.cvtColor(frame_niceR,cv2.COLOR_BGR2GRAY)
        blurR=cv2.GaussianBlur(grayR,(5,5),0)
        ret,thR = cv2.threshold(blurR,100,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        ret,thR = cv2.threshold(thR,127,255,cv2.THRESH_BINARY_INV)
        im,contoursR,hierarchy = cv2.findContours(thR,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contoursL:

            if(cv2.contourArea(cnt)>1000):

                cntL=cnt

        for cnt in contoursR:

            if(cv2.contourArea(cnt)>1000):

                cntR=cnt

        hullL = cv2.convexHull(cntL)
        xL,yL,wL,hL = cv2.boundingRect(cntL)

        hullR = cv2.convexHull(cntR)
        xR,yR,wR,hR = cv2.boundingRect(cntR)
        

        if(objectD=='CUBE'):

            ptsL=p6_algorithmL(hullL,frame_niceL,thL,train)
            ptsR=p6_algorithmR(hullR,frame_niceR,thR,train)

            pts=calZ(ptsL,ptsR,objectD)
            p2=pts[0]
            p3=pts[1]

            p2[0]=round(p2[0]*CubeP2xm+CubeP2xc)
            p2[1]=round(p2[1]*CubeP2ym+CubeP2yc)
            p2[2]=round(p2[2]*CubeP2zm+CubeP2zc)

            p3[0]=round(p3[0]*CubeP3xm+CubeP3xc)
            p3[1]=round(p3[1]*CubeP3ym+CubeP3yc)
            p3[2]=round(p3[2]*CubeP3zm+CubeP3zc)

            thetaY=m.atan((p3[2]-p2[2])/(p3[0]-p2[0]))
            thetaY=round(thetaY*180/m.pi)

            centerCoords(p2,p3,objectD)

            if((train==True) or (test==True)):
 
                labelStatus.config(text='STATUS: '+objectD+' Detected | P2='+str(p2)+' P3='+str(p3)+' thetaY='+str(thetaY)+' | Enter Real Coordinates of P2,P3 (x,y,z):')
            
            else:

                labelStatus.config(text='STATUS: '+objectD+' Detected | P2='+str(p2)+' P3='+str(p3)+' thetaY='+str(thetaY))

        elif(objectD=='CYLINDER'):

            ptsL=p4_algorithmL(hullL,frame_niceL,thL,train)
            ptsR=p4_algorithmR(hullR,frame_niceR,thR,train)

            pts=calZ(ptsL,ptsR,objectD)
            p1=pts[0]
            p2=pts[1]
            p3=pts[2]
            p4=pts[3]

            #print(CylinderP2zm,CylinderP2zc)
            p2[0]=round(p2[0]*CylinderP2xm+CylinderP2xc)
            p2[1]=round(p2[1]*CylinderP2ym+CylinderP2yc)
            p2[2]=round(p2[2]*CylinderP2zm+CylinderP2zc)

            p3[0]=round(p3[0]*CylinderP3xm+CylinderP3xc)
            p3[1]=round(p3[1]*CylinderP3ym+CylinderP3yc)
            p3[2]=round(p3[2]*CylinderP3zm+CylinderP3zc)

            p4[0]=round(p4[0]*CylinderP4xm+CylinderP4xc)
            p4[1]=round(p4[1]*CylinderP4ym+CylinderP4yc)
            p4[2]=round(p4[2]*CylinderP4zm+CylinderP4zc)
            
            p1[0]=round(p1[0]*CylinderP1xm+CylinderP1xc)
            p1[1]=round(p1[1]*CylinderP1ym+CylinderP1yc)
            p1[2]=round(p1[2]*CylinderP1zm+CylinderP1zc)
            
            centerCoords(p1,p4,objectD)
            
            if((train==True) or (test==True)):
 
                labelStatus.config(text='STATUS: '+objectD+' Detected | P1='+str(p1)+' P2='+str(p2)+' p3='+str(p3)+' p4='+str(p4)+' | Enter Real Coordinates of P1,P2,P3,P4 (x,y,z):',font='Helvetica 10 bold',width=120)

            else:

                labelStatus.config(text='STATUS: '+objectD+' Detected | P1='+str(p1)+' P2='+str(p2)+' p3='+str(p3)+' p4='+str(p4))

            
        elif(objectD=='SPHERE'):

            ptsL=p2_algorithmL(hullL,frame_niceL,thL,train)
            ptsR=p2_algorithmR(hullR,frame_niceR,thR,train)

            pts=calZ(ptsL,ptsR,objectD)
            p1=pts[0]
            p2=pts[1]

            #print(pts)

            p1[0]=round(p1[0]*SphereP1xm+SphereP1xc)
            p1[1]=round(p1[1]*SphereP1ym+SphereP1yc)
            p1[2]=round(p1[2]*SphereP1zm+SphereP1zc)

            p2[0]=round(p2[0]*SphereP2xm+SphereP2xc)
            p2[1]=round(p2[1]*SphereP2ym+SphereP2yc)
            p2[2]=round(p2[2]*SphereP2zm+SphereP2zc)

            centerCoords(p1,p2,objectD)

            if((train==True) or (test==True)):
 
                labelStatus.config(text='STATUS: '+objectD+' Detected | P1='+str(p1)+' P2='+str(p2))

            else:

                labelStatus.config(text='STATUS: '+objectD+' Detected | P1='+str(p1)+' P2='+str(p2))


        
        imgL = Image.fromarray(frame_niceL)
        imgR = Image.fromarray(frame_niceR)
        
        imgtkL = ImageTk.PhotoImage(image=imgL)
        imgtkR = ImageTk.PhotoImage(image=imgR)
         
        lmainL.imgtk = imgtkL
        lmainR.imgtk = imgtkR
        
        lmainL.configure(image=imgtkL)
        lmainR.configure(image=imgtkR)

        if(train==True):
            
            tCount+=1

        #window.after(10, main)

        
    else:

        for i in range(0,501,20):

            cv2.line(frameL,(0,i),(700,i),(0,255,0),1)
            cv2.line(frameR,(0,i),(700,i),(0,255,0),1)

        w,h=frameL.shape[:2]
        Left_Stereo_Map= cv2.initUndistortRectifyMap(mtxL,distL,None,OmtxL,(w,h),5)   # cv2.CV_16SC2 this format enables us the programme to work faster
        frame_niceL= cv2.remap(frameL,Left_Stereo_Map[0],Left_Stereo_Map[1], cv2.INTER_LINEAR,0)  # Rectify the image using the kalibration parameters founds during the initialisation
        x,y,w,h=roiL
        frame_niceL=frame_niceL[y:y+h,x:x+w]

        w,h=frameR.shape[:2]
        Right_Stereo_Map= cv2.initUndistortRectifyMap(mtxR,distR,None,OmtxR,(w,h),5)
        frame_niceR= cv2.remap(frameR,Right_Stereo_Map[0],Right_Stereo_Map[1], cv2.INTER_LINEAR,0)
        x,y,w,h=roiR
        frame_niceR=frame_niceR[y:y+h,x:x+w]


        cv2imageL = cv2.cvtColor(frame_niceL, cv2.COLOR_BGR2RGBA)
        cv2imageR = cv2.cvtColor(frame_niceR, cv2.COLOR_BGR2RGBA)

        imgL = Image.fromarray(cv2imageL)
        imgR = Image.fromarray(cv2imageR)
        
        imgtkL = ImageTk.PhotoImage(image=imgL)
        imgtkR = ImageTk.PhotoImage(image=imgR)
         
        lmainL.imgtk = imgtkL
        lmainR.imgtk = imgtkR
        
        lmainL.configure(image=imgtkL)
        lmainR.configure(image=imgtkR)
        
        window.after(100, main)

def calibration():

    global retR,mtxR,distR,rvecsR,tvecsR,hR,wR,OmtxR,roiR
    global retL,mtxL,distL,rvecsL,tvecsL,hL,wL,OmtxL,roiL
    global objpoints  
    global imgpointsR
    global imgpointsL
    global ChessImaR,ChessImaL

    objpoints= []
    imgpointsR= []   
    imgpointsL= []
    
    dFile = open('calibrated_data.txt', 'w')

    criteria =(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 33, 0.001)
    criteria_stereo= (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 33, 0.001)

    objp = np.zeros((7*6,3), np.float32)
    objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

    print('Starting calibration for the 2 cameras... ')

    for i in range(0,31):

        t= str(i)
        ChessImaR= cv2.imread('Calibrating_Data/chessboard-R'+t+'.png',0)   
        ChessImaL= cv2.imread('Calibrating_Data/chessboard-L'+t+'.png',0)   
        retR, cornersR = cv2.findChessboardCorners(ChessImaR,(7,6),None)
        retL, cornersL = cv2.findChessboardCorners(ChessImaL,(7,6),None)

        if (True == retR) & (True == retL):

            objpoints.append(objp)
            cv2.cornerSubPix(ChessImaR,cornersR,(11,11),(-1,-1),criteria)
            cv2.cornerSubPix(ChessImaL,cornersL,(11,11),(-1,-1),criteria)
            imgpointsR.append(cornersR)
            imgpointsL.append(cornersL)


    retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints,imgpointsR,ChessImaR.shape[::-1],None,None)
    hR,wR= ChessImaR.shape[:2]
    OmtxR, roiR= cv2.getOptimalNewCameraMatrix(mtxR,distR,(wR,hR),1,(wR,hR))

    retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints,imgpointsL,ChessImaL.shape[::-1],None,None)
    hL,wL= ChessImaL.shape[:2]
    OmtxL, roiL= cv2.getOptimalNewCameraMatrix(mtxL,distL,(wL,hL),1,(wL,hL))
    print('Calibration done!!')

    dFile.write(str(OmtxR))
    dFile.write(',\n')
    dFile.write(str(OmtxL))
    dFile.write('\n')

    print('Calibration data saved to calibration_data.txt')
        

calibration()
train_model()
main()
window.mainloop()
