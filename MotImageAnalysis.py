########################################
#import
########################################
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import numpy as np
import cv2
import sys

########################################
#functions
########################################
 #calculate image intensity per pixel on ROI region
def intensity(img):
  height, width = img.shape[:2]
  return np.sum(img)/(height*height*width*width)

#img_modify
def img_modify(img,bg):
 img = img -bg
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

########################################
#main
########################################

#initialize
filename =""
argvs = sys.argv  # parameters
argc = len(argvs)  #number of parameters
if (argc !=10):
  print ("please set a parameters")
  quit()
filename = argvs[1]
over_threshold = float(argvs[2])
under_threshold = float(argvs[3])
vmax = float(argvs[4])
bg_img =argvs[5]
x = int(argvs[6])
y = int(argvs[7])
x2 = int(argvs[8])
y2 =int(argvs[9])
argc = len(argvs)  #number of parameters


i=1
over_threshold_num = 0;
under_threshold_num = 0;



bg = cv2.imread(filename+"/Image-"+filename+"-"+str(bg_img)+".tiff",-1)
bg = np.array(bg, dtype=np.float16) 
roi_bg = bg[y:y2,x:x2]
accum_img = np.zeros((128,168),np.float16)
accum_img_under = np.zeros((128,168),np.float16)
accum_roi_img = np.zeros((y2-y,x2-x),np.float16)
accum_roi_img_under = np.zeros((y2-y,x2-x),np.float16)
roi_array = np.array([],np.float16)



#load each images
while(True):
    img = cv2.imread(filename+"/Image-"+filename+"-"+str(i)+".tiff", -1)
    if img is None:
      break
    img = np.array(img, dtype=np.float16) 
    modified_img = img_modify(img,bg)
    modified_roi_img = modified_img[y:y2,x:x2]
    roi = intensity(modified_roi_img)
    roi_array = np.append(roi_array,roi)

    if roi >over_threshold:
      accum_img = accum_img+modified_img
      accum_roi_img = accum_roi_img+modified_roi_img
      print(str(i)+" "+str(roi) +" "+ str(MOT_fluorescence_to_number(np.sum(modified_roi_img),20,15)))
      over_threshold_num += 1

    if roi <under_threshold:
      accum_img_under = accum_img_under+modified_img
      accum_roi_img_under = accum_roi_img_under+modified_roi_img
      under_threshold_num += 1     

    i += 1

accum_img = accum_img/over_threshold_num
accum_roi_img = accum_roi_img/over_threshold_num
accum_img_under = accum_img_under/under_threshold_num
accum_roi_img_under = accum_roi_img_under/under_threshold_num

accum_img_difference = accum_img
accum_roi_img_difference = accum_roi_img

#detect circle
#ret,recon_img = cv2.threshold(accum_img_difference,vmax*0.1,65536,cv2.THRESH_BINARY)
#recon_img = recon_img.astype(np.uint8)
#circles = cv2.HoughCircles(recon_img, cv2.HOUGH_GRADIENT, 1, 60, param1=15, param2=1, minRadius=2, maxRadius=50)
#print(circles)

print("over-threshold:"+str(over_threshold_num)+"under-threshold:"+str(under_threshold_num))


fig = plt.figure(figsize=(22,12))

# サブプロットを追加
ax1 = fig.add_subplot(3,3,1)
ax2 = fig.add_subplot(3,3,2)
ax3 = fig.add_subplot(3,3,3)
ax4 = fig.add_subplot(3,3,4)
ax5 = fig.add_subplot(3,3,5)
ax6 = fig.add_subplot(3,3,6)
ax7 = fig.add_subplot(3,3,7)
ax8 = fig.add_subplot(3,3,8)
ax9 = fig.add_subplot(3,3,9)
#draw intensity graph
ax1.set_title("time evolution of intensity")
ax1.axhline(y=over_threshold, xmin=0, xmax=i, linewidth=2, color = 'salmon')
ax1.axhline(y=under_threshold, xmin=0, xmax=i, linewidth=2, color = 'green')
#ax1.axhline(x=int(bg_img), xmin=0, xmax=i, linewidth=2, color = 'green')
ax1.plot(roi_array)


#draw cross-section image
ax2.set_title("cross-section image")
ax2.plot(accum_img[int((y+y2)/2)], color = 'salmon')
ax2.plot(accum_img_under[int((y+y2)/2)], color = 'green')


#draw cross-section image of ROI
ax3.set_title("Fourier")
ax3.plot(np.fft.fft(roi_array))
ax3.set_xlim([0, 500])


#draw 2d color map
ax4.set_title("2d color map (over threshold)")
cm_img= ax4.imshow(accum_img, vmin = 0, vmax = vmax)
rect = patches.Rectangle((x,y),x2-x,y2-y,linewidth=1,edgecolor='r',facecolor='none')
ax4.add_patch(rect)
plt.colorbar(cm_img, ax=ax4)

#draw recognized circles
#for j in circles[0,:]:
#    ax4.add_patch(patches.Rectangle((j[0]-j[2]/2,j[0]-j[2]/2),2*j[2],2*j[2],linewidth=1,edgecolor='g',facecolor='none'))
#plt.colorbar(cm_img, ax=ax4)


#draw cross-section image of ROI
ax5.set_title("2d color map (under threshold)")
cm_img_under= ax5.imshow(accum_img_under, vmin = 0, vmax = vmax)
ax5.add_patch(rect)
plt.colorbar(cm_img_under, ax=ax5)


#draw 2d color map difference
ax6.set_title("2d color map (difference)")
cm_img_difference = ax6.imshow(accum_img_difference, vmin = 0, vmax = vmax)
plt.colorbar(cm_img_difference , ax=ax6)



#draw cross-section image of ROI
ax7.set_title("2d color map of ROI (over threshold)")
cm_roi_img= ax7.imshow(accum_roi_img, vmin = 0, vmax = vmax)
plt.colorbar(cm_roi_img, ax=ax7)



#draw cross-section image of ROI
ax8.set_title("2d color map of ROI (under threshold)")
cm_roi_img_under= ax8.imshow(accum_roi_img_under, vmin = 0, vmax = vmax)
plt.colorbar(cm_roi_img_under, ax=ax8)

#draw 2d color map of ROI difference
ax9.set_title("2d color map (difference)")
cm_roi_img_difference = ax9.imshow(accum_roi_img_difference, vmin = 0, vmax = vmax )
plt.colorbar(cm_roi_img_difference, ax=ax9)



plt.show()

#save image
#cv2.imwrite("accum_img-"+filename+".tif",accum_img)
fig.savefig("summary-"+filename+".png")


#terminate
cv2.destroyAllWindows()

