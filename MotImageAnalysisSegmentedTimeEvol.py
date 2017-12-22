########################################################################################################################
#import
########################################################################################################################
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import numpy as np
import cv2
import sys
import matplotlib.colors
from matplotlib.widgets import Slider, Button, RadioButtons
import matplotlib.cm as cm
########################################################################################################################
#functions
########################################################################################################################

 #calculate image intensity per pixel on ROI region
def intensity(img):
  height, width = img.shape[:2]
  return np.sum(img)/(height*width)

#img_modify
def img_modify(img):
# img=cv2.medianBlur(img,5)
 return img

def MOT_fluorescence_to_number(gross,detuning,Laser):
  h = 0.6626
  frequency = 384.2
  distance = 0.04
  radii = 12.8
  r = 0.5
  Isat = 1.67
  pi = 3.141592
  detuning = 2*pi*detuning
  nw = 2*pi*6.0666
  power = 9.1e-10*gross

  energy = h*frequency
  photon_PD = power/energy
  omega=radii*radii/(4*distance*distance)
  photon_MOT = photon_PD / omega
  I=Laser*6/(2*3.14*r*r)
  s=(I/Isat)/(1+(2*detuning/nw)*(2*detuning/nw))
  rate=nw*s/(1+s)/2
  number=photon_MOT/rate*1.0E+15
  return number


class roi_analysis:
      def __init__(self,t1,t2,x,y,x2,y2):
        self.t1 = t1
        self.t2 = t2
        self.x = x
        self.y = y
        self.x2 = x2
        self.y2 = y2
        accum_img = np.zeros((128,168),np.float64)
        roi_array = np.array([],np.float64)
        i = 0
        while(True):
          img = cv2.imread(filename+"/Image-"+filename+"-"+str(i)+".tiff", -1)
          if img is None:
            break
          img = np.array(img, dtype=np.float64) 
          modified_img = img_modify(img)
          modified_roi_img = modified_img[y:y2,x:x2]
          roi = intensity(modified_roi_img)
          roi_array = np.append(roi_array,roi)
          accum_img = accum_img+modified_img    
          i += 1
        accum_img = accum_img/i
        self.accum_img = accum_img
        self.roi_array = roi_array

########################################################################################################################
#draw fig
########################################################################################################################
      def plot(self):

        fig = plt.figure(figsize=(15,6))
        plt.subplots_adjust(left=0.1, right=0.9, top=0.8, bottom=0.2)
        suptitle = plt.suptitle("summary-"+filename+".png, ROI: x="+str(self.x)+",y="+str(self.y)+",x2="+str(self.x2)+",y2="+str(self.y2) , x = 0.5, y = 0.97, fontsize=24)
        # サブプロットを追加
        ax1 = fig.add_subplot(1,2,1)
        ax2 = fig.add_subplot(1,2,2)
        #draw intensity graph
        ax1.set_title("time evolution of intensity")

        ax1.axvspan(0.001, self.t1, facecolor='black', alpha=0.5)
        if self.t2 != 0:
                ax1.axvspan(self.t2, i, facecolor='black', alpha=0.5)

        ax1.plot(self.roi_array , color ="b", linewidth=1)
        ax1_data1 = ax1.twinx()
        ax1_data2 = ax1.twinx()

        ax1_data1.plot(dat1 , color ="g", linewidth=0.5)
        ax1_data2.plot(dat2 , color ="r", linewidth=0.5)

        #draw 2d color map
        ax2.set_title("over threthold (num. of images: "+str(i) +")")
        cm_img= ax2.imshow(self.accum_img, vmin = vmin0, vmax = vmax0,cmap=cm.jet, interpolation='bicubic')
        rect = patches.Rectangle((self.x,self.y),self.x2-self.x,self.y2-self.y,linewidth=1,edgecolor='blue',facecolor='none')
        ax2.add_patch(rect)
        plt.colorbar(cm_img, ax=ax2) 


        #slider
        axmin = fig.add_axes([0.05, 0.02, 0.4, 0.02], facecolor="gray")
        axmax  = fig.add_axes([0.05, 0.05, 0.4, 0.02], facecolor="gray")

        smin = Slider(axmin, 'Min', 0, 10, valinit=vmin0)
        smax = Slider(axmax, 'Max', 0, 300, valinit=vmax0)

        def update(val):
                cm_img.set_clim([smin.val,smax.val])
                fig.canvas.update()
        smin.on_changed(update)
        smax.on_changed(update)


        fig.savefig("./"+filename+"/summary-segmentedTimeEvol-"+filename+"-range[x="+str(self.x)+"-"+str(self.x2)+",y="+str(self.y)+"-"+str(self.y2)+"].png")
        plt.close(1)

########################################################################################################################
#save images
########################################################################################################################



########################################################################################################################
#initialize
########################################################################################################################
#parameters
vmin0 = 0
vmax0 = 300
#initialize
filename =""
argvs = sys.argv  # parameters
argc = len(argvs)  #number of parameters
if (argc !=10):
  print ("please set a parameters")
  quit()
filename = argvs[1]
t1 = int(argvs[2])
t2 = int(argvs[3])
x = int(argvs[4])
y = int(argvs[5])
dx = int(argvs[6])
dy =int(argvs[7])
x_num = int(argvs[8])
y_num =int(argvs[9])


i=1

accum_img = np.zeros((128,168),np.float64)
roi_array = np.array([],np.float64)

########################################################################################################################
#load each images
########################################################################################################################
##loading Dat File
dat1 =[]
dat2 =[]
for l in open(filename+"/Graph-"+filename+".dat").readlines():
    data = l[:-1].split('\t')
    dat1 += [float(data[9])]
    dat2 += [float(data[10])]
roi = []
segment_num = 0
for i in range (0,x_num):
  for j in range (0,y_num):
    print(j)
    roi.append(roi_analysis(t1,t2,x+i*dx,y+j*dx,x+(i+1)*dx,y+(j+1)*dx))
    roi[segment_num].plot()
    segment_num += 1



########################################################################################################################
#draw fig
########################################################################################################################


########################################################################################################################
#save images
########################################################################################################################



#terminate
cv2.destroyAllWindows()

