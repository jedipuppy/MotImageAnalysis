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
def intensity(img,x,y,x2,y2):
  roi = img[x:x2,y:y2]
  return np.sum(roi)/((x2-x)*(y2-y))
#img_modify
def img_modify(img,bg):
 cv2.subtract(img,bg,img)
 img=cv2.medianBlur(img,5)
 return img



########################################
#main
########################################

#initialize
filename =""
argvs = sys.argv  # parameters
argc = len(argvs)  #number of parameters
if (argc !=7):
  print ("please set a  threshold")
  quit()
filename = argvs[1]
threshold = int(argvs[2])
bg_img =argvs[3]
x = int(argvs[4])
y = int(argvs[5])
x2 = int(argvs[6])
y2 =int(argvs[7])
argc = len(argvs)  #number of parameters


bg = cv2.imread(filename+str(bg_img)+".tif",-1)
i=1
accum_img = bg
cm_accum_img = bg
roi_array = np.array([])






#load each images
while(True):
    img = cv2.imread(filename+str(i)+".tif", -1)
    if img is None:
      break
    modified_img = img_modify(img,bg)
    roi = intensity(modified_img,x,y,x2,y2)
    roi_array = np.append(roi_array,roi)
    if roi >threshold:
      accum_img =cv2.add(accum_img,modified_img)
      print(str(roi) +" "+ str(np.sum(accum_img)))
    c = cv2.waitKey(50) & 0xFF
    if c==27: # ESC
        break
    i += 1

recon_img = accum_img.astype(np.uint8)
circles = cv2.HoughCircles(recon_img, cv2.HOUGH_GRADIENT, 1, 60, param1=15, param2=1, minRadius=2, maxRadius=50)
#circles = np.uint16(np.around(circles))
print(circles)

#terminate
cv2.waitKey(2000)

	#plot intensity

fig, (axL,axC, axR) = plt.subplots(ncols=3, figsize=(22,4))

#draw intensity graph
axL.axhline(y=threshold, xmin=0, xmax=i, linewidth=2, color = 'salmon')
axL.plot(roi_array)


#draw cross-section image
axC.plot(accum_img[(y+y2)/2])
#draw 2d color map
cm_img= axR.imshow(accum_img)
rect = patches.Rectangle((x,y),x2-x,y2-y,linewidth=1,edgecolor='r',facecolor='none')
axR.add_patch(rect)

#draw recognized circles
for j in circles[0,:]:
    axR.add_patch(patches.Rectangle((j[0]-j[2]/2,j[0]-j[2]/2),2*j[2],2*j[2],linewidth=1,edgecolor='g',facecolor='none'))


plt.colorbar(cm_img)


#save image
cv2.imwrite("accum_img-"+filename+".tif",accum_img)
fig.savefig("summary-"+filename+".png")


plt.show()
#terminate
cv2.destroyAllWindows()

