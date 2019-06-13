#Next steps
#start Buliding in logic 
#see the Paper notes
#Make Change->Save->TickToCommitt->Push


import cv2
import numpy as np
import sys

#Parameters
history = 50
threshold = 100
shadow = False
area = 20

# Video Capture  from webcam
# capture = cv2.VideoCapture(0)
capture = cv2.VideoCapture(r"C:\Users\Steve\Desktop\VideoFiles\PowerPTAnimations\Blocks.mp4")

#this is is where the comparision of current frame and model occurs
#Important to note that this is for each pixel (not the whole frame
fgbg = cv2.createBackgroundSubtractorMOG2(history, threshold, shadow)

# Keeps track of what frame we're on
frameCount = 0

while(1):
	# Return Value and the current frame
    #return determines if a frame is detected.
    #This is important to prevent errors when a clip finishes (no more frames detected).
    #if a frame is NOT detected we break from the loop and finish. 
    ret, frame = capture.read()
	#  Check if a current frame actually exist
    if not ret:
        break
    #increment the frame by one.            
    frameCount += 1
	# Resize the frame (50% on X and Y)
    #makes more manageable
    resizedFrame = cv2.resize(frame, (0, 0), fx=0.50, fy=0.50)
#    resizedFrame = frame

	# Get the foreground mask (fg)
#   fgmask is a black and whote frame
    fgmask = fgbg.apply(resizedFrame)

	# Count all the non zero pixels within the mask
    #non-zero is a white pixel - i.e once that quantifies movement.
    count = np.count_nonzero(fgmask)
        
#    print('Frame: %d, Pixel Count: %d' % (frameCount, count))

	# Determine how many pixels do you want to detect to be considered "movement" or change
    #can't run this on the first frame, as there is no Previous frame, so there will be lots of noise
    #as first image will be by default solid black.
    
    #remember count is how many pixels have changed (based on the model) 
    if (frameCount > 1 and count > 1000):
#    if frameCount > 1: 
        cv2.putText(resizedFrame, "Frame #: " + str(frameCount), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        (im2, contours, hierarchy) = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if cv2.contourArea(c) > area:
                (x, y, w, h) = cv2.boundingRect(c)
 #               print(cv2.contourArea(c))

        #draw bounding box
                cv2.rectangle(resizedFrame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#                print(x,y,w,h)
 

        cv2.imshow('Frame', resizedFrame)
        cv2.imshow('Mask', fgmask)


        k = cv2.waitKey(100) & 0xff
        if k == 27:
            break

capture.release()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:




