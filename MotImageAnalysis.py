import sys

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import random
import cv2
import matplotlib.gridspec as gridspec
import re
from matplotlib.widgets import Slider
import matplotlib.cm as cm
import matplotlib.patches as patches

class ImageAnalysis():
    def __init__(self,filename,upper_threshold,lower_threshold,background_file_number,start_number,stop_number,x,x2,y,y2):
        self.upper_img_num = 0
        self.lower_img_num = 0
        self.x = x
        self.x2 = x2
        self.y =y
        self.y2 = y2
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold
        self.background_file_number = background_file_number
        self.start_number = start_number
        self.stop_number =stop_number
        self.filename = filename
        self.loadData()
        self.update()



    def update(self):
        self.img_array = []
        self.roi_array = []
        self.upper_img = np.zeros((128,168),np.float64)
        self.lower_img = np.zeros((128,168),np.float64)
        self.bg_img = np.zeros((128,168),np.int8)
        self.diff_img = np.zeros((128,168),np.int8)       
        self.loadImgAll(self.filename)

    def loadData(self):
        self.data1 = []
        self.data2 = []
        self.data3 = []
        self.data4 = []        
        self.data5 = []
        self.data6 = []
        self.data7 = [] 
        self.data8 = [] 
        print(self.filename+"/Graph-"+self.filename+".dat")
        for l in open(self.filename+"/Graph-"+self.filename+".dat").readlines():
            data = l[:-1].split('\t')   
            self.data1 += [float(data[5])]
            self.data2 += [float(data[6])]
            self.data3 += [float(data[7])]
            self.data4 += [float(data[8])]            
            self.data5 += [float(data[9])]
            self.data6 += [float(data[10])]
            self.data7 += [float(data[11])]      
            self.data8 += [float(data[11])] 

  #ファイル番号を入れるとその画像ファイルを返す関数
    def loadImg(self,filename,file_number):

        img = cv2.imread(filename+"/Image-"+filename+"-"+str(file_number)+".tiff", -1)
        if img is None:
            return 1;
        modified_img = self.modifyImg(img,self.bg_img)

        roi_modified_img =modified_img[self.y:self.y2,self.x:self.x2]



        roi = self.intensity(roi_modified_img)
        self.img_array.append(modified_img)
        self.roi_array.append(roi)
        return 0;

    def loadBG(self,filename,file_number):
        img = cv2.imread(filename+"/Image-"+filename+"-"+str(file_number)+".tiff", -1)
        print("bg",filename+"/Image-"+filename+"-"+str(file_number)+".tiff")
        if img is None:
            return np.zeros((128,168),np.float64);
        return img;


  #画像をいれると処理を返してくれる
    def modifyImg(self,img,bg):

        modified_img = cv2.subtract(img,bg)
        return modified_img;

  #強度を返す
    def intensity(self,img):
        height, width = img.shape[:2]
        return np.sum(img)/(height*width)

  #画像を蓄積
    def accum_img(self,img,roi):
        if roi > self.upper_threshold:
            self.upper_img_num += 1         
            self.upper_img += img
        if roi < self.lower_threshold:
            self.lower_img_num += 1       
            self.lower_img += img        



#連番で画像を読み込む関数
    def loadImgAll(self,filename):
        print("BG file number",self.background_file_number)
        self.bg_img = self.loadBG(filename,self.background_file_number)
        file_number = 0
        while(True):
            loadFlag = self.loadImg(filename,file_number)
            if loadFlag is 1:
                break;
            file_number += 1
            print(file_number)

            if file_number > start_number and file_number <stop_number and stop_number != 0:
                self.accum_img(self.img_array[-1],self.roi_array[-1])
        if self.upper_img_num != 0:
            print ("accum upper",file_number)
            self.upper_img =  self.upper_img/self.upper_img_num
        if self.lower_img_num != 0:
            print ("accum lower",file_number)
            self.lower_img = self.lower_img/self.lower_img_num

            self.diff_img = self.upper_img - self.lower_img


########################################################################################################################
#input parameters
########################################################################################################################

argvs = sys.argv  # parameters
argc = len(argvs)  #number of parameters
if (argc !=11):
  print ("please set a parameters")
  quit()
filename = argvs[1]
upper_threshold = float(argvs[2])
lower_threshold = float(argvs[3])
start_number = int(argvs[4])
stop_number = int(argvs[5])
background_file_number =argvs[6]
x = int(argvs[7])
y = int(argvs[8])
x2 = int(argvs[9])
y2 =int(argvs[10])
argc = len(argvs)  #number of parameters


########################################################################################################################
#draw fig
########################################################################################################################
vmin0 = 0
vmax0 = 20

image_ana = ImageAnalysis(filename,upper_threshold,lower_threshold,background_file_number,start_number,stop_number,x,x2,y,y2)
fig = plt.figure(figsize=(14,9))
fig.subplots_adjust(top = 0.930,bottom =0.095,left=0.060,right =0.940)
suptitle = plt.suptitle("summary-"+filename+".png, ROI: x="+str(x)+",y="+str(y)+",x2="+str(x2)+",y2="+str(y2) , x = 0.5, y = 0.97, fontsize=18)
gs = gridspec.GridSpec(3, 3,
                       width_ratios=[1, 1,1],
                       height_ratios=[1.5,1,3],
                       wspace=0.1,
                       hspace=0.2
                       )
ax1 = plt.subplot(gs[0,:])
ax3 = plt.subplot(gs[1,:])
ax4 = plt.subplot(gs[2,0])
ax5 = plt.subplot(gs[2,1])
ax6 = plt.subplot(gs[2,2])


ax1.plot(image_ana.roi_array, linewidth=2)

ax1.axvspan(0, image_ana.start_number, facecolor='black', alpha=0.5)
if image_ana.stop_number != 0:
    ax1.axvspan(image_ana.stop_number,len(image_ana.data1) , facecolor='black', alpha=0.5)

ax1.axhline(y=image_ana.upper_threshold, linewidth=1, color = 'green')
ax1.axhline(y=image_ana.lower_threshold, linewidth=1, color = 'green') 
ax1.axvline(x =int(background_file_number), linewidth=1, color = 'salmon')        
data5_ax1 = ax1.twinx()
data5_ax1.plot(image_ana.data5 , linewidth=0.8, color ="r", label="B field")
data6_ax1 = ax1.twinx()
data6_ax1.plot(image_ana.data6 , linewidth=0.8, color ="g", label="Y target temp")

data1_ax3 = ax3.twinx()
data1_ax3.plot(image_ana.data1, linewidth=0.8, color ="c", label="I2")
data2_ax3 = ax3.twinx()
data2_ax3.plot(image_ana.data2 , linewidth=0.4, color ="b", label="scan signal")
data3_ax3 = ax3.twinx()
data3_ax3.plot(image_ana.data3 , linewidth=0.8, color ="g", label="resonator")
data7_ax3 = ax3.twinx()
data7_ax3.plot(image_ana.data7 , linewidth=0.8, color ="r", label="wavelength TOF")
data8_ax3 = ax3.twinx()
data8_ax3.plot(image_ana.data8 , linewidth=0.8, color ="k", label="wavelength Laser")

ax3.legend(loc=0)
data1_ax3.legend(loc=1)
data2_ax3.legend(loc=2)
data3_ax3.legend(loc=3)
data7_ax3.legend(loc=4)
data8_ax3.legend(loc=5)

data5_ax1.legend(loc=0)
data6_ax1.legend(loc=1)

upper_img_fig = ax4.imshow(image_ana.upper_img, vmin = vmin0, vmax = vmax0,cmap=cm.jet, interpolation='nearest')
lower_img_fig = ax5.imshow(image_ana.lower_img, vmin = vmin0, vmax = vmax0,cmap=cm.jet, interpolation='nearest')
diff_img_fig = ax6.imshow(image_ana.diff_img, vmin = vmin0, vmax = vmax0,cmap=cm.jet, interpolation='nearest')

rect1 = patches.Rectangle((x,y),x2-x,y2-y,linewidth=1,edgecolor='w',facecolor='none')
rect2 = patches.Rectangle((x,y),x2-x,y2-y,linewidth=1,edgecolor='w',facecolor='none')
rect3 = patches.Rectangle((x,y),x2-x,y2-y,linewidth=1,edgecolor='w',facecolor='none')
ax4.add_patch(rect1)
ax5.add_patch(rect2)
ax6.add_patch(rect3)
########################################################################################################################
#slider
########################################################################################################################
axmin = fig.add_axes([0.05, 0.02, 0.4, 0.02], facecolor="gray")
axmax  = fig.add_axes([0.05, 0.05, 0.4, 0.02], facecolor="gray")

smin = Slider(axmin, 'Min', 0, 10, valinit=vmin0)
smax = Slider(axmax, 'Max', 0, 100, valinit=vmax0)

def update(val):
    upper_img_fig.set_clim([smin.val,smax.val])
    lower_img_fig.set_clim([smin.val,smax.val])
    diff_img_fig.set_clim([smin.val,smax.val])


    fig.canvas.update()
smin.on_changed(update)
smax.on_changed(update)




########################################################################################################################
#save images
########################################################################################################################
cv2.imwrite("./"+filename+"/upper_img-"+filename+".tif",np.array(np.round(image_ana.upper_img), dtype=np.int16) )
cv2.imwrite("./"+filename+"/lower_img-"+filename+".tif",np.array(np.round(image_ana.lower_img), dtype=np.int16))
cv2.imwrite("./"+filename+"/diff_img"+filename+".tif",np.array(np.round(image_ana.diff_img), dtype=np.int16))
fig.savefig("./summary-"+filename+".png")

plt.show()

