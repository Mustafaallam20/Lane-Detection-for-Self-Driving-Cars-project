#!/usr/bin/env python
# coding: utf-8

# import some libraries 
# 

# In[1]:


import numpy as np
import time
import cv2 
import glob
import os
import matplotlib.pyplot as plt


#  Load yolo weights and cfg
#  

# In[2]:


weights_path = os.path.join("D:/2nd term/Digital image processing/Project","yolov3.weights")
config_path = os.path.join("D:/2nd term/Digital image processing/Project","yolov3.cfg")
print("loaded!")


# Load Neural net in cv2
# 

# In[3]:



net = cv2.dnn.readNetFromDarknet(config_path,weights_path)


# Get Layers names
# 

# In[4]:



names = net.getLayerNames() 
names


# load the test image

# In[5]:


image_path = os.path.join("D:/2nd term/Digital image processing/Project","tst_img.png")
img = cv2.imread(image_path)
img_conv = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_conv)


# In[6]:


(H, W) = img.shape[ :2]
layers_names = [names[i - 1] for i in net.getUnconnectedOutLayers()]


# In[7]:


layers_names


# Run the inferences on the test image

# In[8]:


blob = cv2.dnn.blobFromImage(img_conv, 1/255.0,(416,416), crop = False , swapRB = False)
net.setInput(blob)

#calc runtime algo
start_t = time.time()
layers_output = net.forward(layers_names)
print("A forward pass through yolov3 took{}".format(time.time() - start_t))


# In[9]:


layers_output


# In[10]:


boxes = []
confidences = []
classIDs = []


# In[11]:


for output in layers_output:
  for detection in output:
      scores = detection[5:]
      classID = np.argmax(scores)
      confidence = scores[classID]
      
      if (confidence > 0.85):
          box = detection[:4]*np.array([W ,H,W ,H])
          bx , by , bw , bh = box.astype("int")
          
          x = int(bx - (bw/2))
          y = int(by - (bh/2))
          
          boxes.append([x , y , int(bw), int(bh)])
          confidences.append(float(confidence))
          classIDs.append(classID)
      


# In[12]:


idxs = cv2.dnn.NMSBoxes(boxes , confidences , 0.8 , 0.8)


# Reload Labels file
# 

# In[13]:


labels_path = os.path.join("D:/2nd term/Digital image processing/Project","coco.names")
labels = open(labels_path).read().strip().split("\n")


# In[14]:


labels


# Plot the bounding Boxes in the image

# In[15]:


for i in idxs.flatten():
    (x , y) = [boxes[i][0] , boxes[i][1]]
    (w , h) = [boxes[i][2] , boxes[i][3]]
    
    cv2.rectangle(img_conv, (x,y), (x + w ,y + h), (0 , 255 , 255, 2))
    cv2.putText(img, "{}: {}".format(labels[classIDs[i]], confidences[i]) ,
        (x, y-5), cv2.FONT_HERSHEY_SIMPLEX,  0.5 , (0 , 0 , 255) , 2 )


# Plotting resulting image

# In[ ]:


cv2.imshow("detected cars", cv2.cvtColor(img_conv, cv2.COLOR_RGB2BGR))
    
cv2.waitKey(0)


# In[ ]:




