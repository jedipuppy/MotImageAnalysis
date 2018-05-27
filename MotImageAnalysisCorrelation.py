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
#functions,clasees
########################################################################################################################

 #calculate image intensity per pixel on ROI region
def intensity(img):
  height, width = img.shape[:2]
  return np.sum(img)/(height*width)

#img_modify
def img_modify(img):
# img=cv2.medianBlur(img,5)
 return img


class roi_analysis:
      def __init__(self,t1,t2,x,y,x2,y2):
        self.t1 = t1
        self.t2 = t2
        self.x = x
        self.y = y
        self.x2 = x2
        self.y2 = y2
        self.accum_img = np.zeros((128,168),np.float64)
        self.roi_array = np.array([],np.float64)


      def load_image(self,img):
          img = np.array(img, dtype=np.float64) 
          modified_img = img_modify(img)
          modified_roi_img = modified_img[self.y:self.y2,self.x:self.x2]
          roi = intensity(modified_roi_img)
          self.roi_array = np.append(self.roi_array,roi)
          self.accum_img = self.accum_img+modified_img    



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
if (argc !=13):
  print ("please set a parameters")
  quit()
filename = argvs[1]
t1 = int(argvs[2])
t2 = int(argvs[3])
x = int(argvs[4])
y = int(argvs[5])
dx = int(argvs[6])
dy =int(argvs[7])
num_of_segments =int(argvs[8])

i=1
control_x = int(argvs[9])
control_y = int(argvs[10])
control_x2 = int(argvs[11])
control_y2 = int(argvs[12])
accum_img = np.zeros((128,168),np.float64)
roi_array = np.array([],np.float64)

########################################################################################################################
#load each images
########################################################################################################################
##loading Dat File
dat =[]
for l in open(filename+"/Graph-"+filename+".dat").readlines():
    data = l[:-1].split('\t')
    dat += [float(data[5])]

roi = [[]]
control_segment = roi_analysis(t1,t2,control_x,control_y,control_x2,control_y2)
for j in range (0,num_of_segments):
  roi.append([])
  for i in range (0,num_of_segments):
    print(j)
    roi[j].append(roi_analysis(t1,t2,x+i*dx,y+j*dx,x+(i+1)*dx,y+(j+1)*dx))


img_num = t1
while(img_num < t2 or t2 == 0):
  print(img_num)
  img = cv2.imread(filename+"/Image-"+filename+"-"+str(img_num)+".tiff", -1)
  if img is None:
    break
  control_segment.load_image(img)
  for j in range (0,num_of_segments):
    for i in range (0,num_of_segments):
        roi[j][i].load_image(img)
  img_num +=1

corr_array = np.zeros((num_of_segments,num_of_segments))
segment_num = 0
for j in range (0,num_of_segments):
  for i in range (0,num_of_segments):
    corr = np.corrcoef(roi[j][i].roi_array, control_segment.roi_array)
    print(str(j)+" "+str(i)+" "+str(corr[0,1]))
    corr_array[j][i] = corr[0,1]


########################################################################################################################
#draw fig
########################################################################################################################
fig = plt.figure(figsize=(18,5))
fig.subplots_adjust(hspace=.4)
suptitle = plt.suptitle("correlation-"+filename+".png, ROI: x="+str(control_x)+",y="+str(control_y)+",x2="+str(control_x2)+",y2="+str(control_y2) , x = 0.5, y = 0.97, fontsize=18)
ax1 = fig.add_subplot(1,3,1)
ax2 = fig.add_subplot(1,3,2)
ax3 = fig.add_subplot(1,3,3)
control_fig = ax1.imshow(control_segment.accum_img)
plt.colorbar(control_fig, ax=ax1)



for j in range (0,num_of_segments):
  for i in range (0,num_of_segments):
    rect = patches.Rectangle((x+j*dx,y+i*dy),dx,dy,linewidth=1,edgecolor='r',facecolor='none')
    ax1.add_patch(rect)

rect = patches.Rectangle((control_x,control_y),control_x2 - control_x,control_y2 - control_y,linewidth=1.5,edgecolor='g',facecolor='none')
ax1.add_patch(rect)

corr_fig = ax2.imshow(corr_array, vmin = -0.7, vmax = 0.7,cmap=cm.jet, interpolation='nearest')
plt.colorbar(corr_fig, ax=ax2)

ax3.plot(corr_array)



########################################################################################################################
#slider
########################################################################################################################
axmin = fig.add_axes([0.05, 0.02, 0.4, 0.02], axisbg="gray")
axmax  = fig.add_axes([0.05, 0.05, 0.4, 0.02], axisbg="gray")

smin = Slider(axmin, 'Min', 0, 10000, valinit=0)
smax = Slider(axmax, 'Max', 0, 1000000, valinit=100000)

def update(val):
    control_fig.set_clim([smin.val,smax.val])
    fig.canvas.update()
smin.on_changed(update)
smax.on_changed(update)


plt.show()


########################################################################################################################
#save images
########################################################################################################################
fig.savefig("./correlation-"+filename+"x="+str(control_x)+"x2="+str(control_x2)+"y="+str(control_y)+"y2="+str(control_y2)+".png")


#terminate
cv2.destroyAllWindows()

