import cv2
import numpy as np
import copy

cap = cv2.VideoCapture('project_video.mp4')
straight = cv2.imread('straight.jpg')

r_start = 0
r_end = 0

l_start = 0
l_end = 0
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
# Check if camera opened successfully
while(cap.isOpened()):
    
  # Capture frame-by-frame
  ret, frame = cap.read()

  
  if ret == True:
      
        frame2 = copy.deepcopy(frame)
        frame1 = copy.deepcopy(frame)
        
        width = int(frame.shape[1])
        height = int(frame.shape[0])
        
        height_max = height-50
        height_min = int(height/1.55)
        
        mask = np.zeros(frame.shape, dtype = "uint8")
        cv2.rectangle(mask, (0, 425), (1500, 740), (255, 255, 255), -1)
        maskedImg = cv2.bitwise_and(frame, mask)
        hsl = cv2.cvtColor(maskedImg, cv2.COLOR_BGR2HLS)
        
        low_yellow = np.array([15,38,115])
        upper_yellow = np.array([35,204,255])
        mask_yellow = cv2.inRange(hsl,low_yellow,upper_yellow)
        
        low_white = np.array([0,200,0])
        upper_white = np.array([180,255,255])
        mask_white = cv2.inRange(hsl,low_white,upper_white)
        
        hsl_mask = cv2.bitwise_or(mask_yellow,mask_white)
        hsl_combined = cv2.bitwise_and(frame, frame, mask = hsl_mask)
        
        gray = cv2.cvtColor(hsl_combined,cv2.COLOR_BGR2GRAY)
        
        blurr = cv2.GaussianBlur(gray, (5, 5), 0)
        blurr=cv2.medianBlur(gray,5)

        edges = cv2.Canny(blurr,50,100)
        width = frame.shape[1]
        height = frame.shape[0]
        pts = np.array([[0,height],[width, height],[width*.62, height/1.5],[width*.4, height/1.5]])
        cv2.fillPoly(mask, np.int_([pts]), (255,255,255))
        lines = cv2.HoughLinesP(edges, cv2.HOUGH_PROBABILISTIC, np.pi/180,100, minLineLength=5,maxLineGap=100)

        leftx = []
        lefty = []
        
        rightx = []
        righty = []
        
        rs_avg = []
        re_avg = []
        
        ls_avg = []
        le_avg = []
        
        
        if lines is not None:
            count = 0
            for line in lines:
                
                count+=1
                x1, y1, x2, y2 = line[0]
                m=(y2-y1)/(x2-x1)
                b=y1-m*x1
                
                if(x1<1100) and (x2<1100):
                      if  abs(m)>=0.5:
                            if b>800 or b<200:
                 
                                if m<=0:
                                    
                                    leftx.extend([x1, x2])
                                    lefty.extend([y1, y2])
                                    leftLine = np.poly1d(np.polyfit(lefty, leftx, deg = 1))
                                    l_start = int(leftLine(height_max))
                                    ls_avg.append(l_start)
                                    l_end = int(leftLine(height_min))
                                    le_avg.append(l_end)

                     
                                else:
                                    rightx.extend([x1, x2])
                                    righty.extend([y1, y2])
                                    rightLine = np.poly1d(np.polyfit(righty, rightx, deg = 1))
                                    r_start = int(rightLine(height_max))
                                    rs_avg.append(r_start)
                                    r_end = int(rightLine(height_min))
                                    re_avg.append(r_end)
                                  
        alpha = .3
        r_avg_start = np.mean(rs_avg)
        r_avg_end = np.mean(re_avg)
        l_avg_start = np.mean(ls_avg)
        l_avg_end = np.mean(le_avg)
        
        if np.isnan(r_avg_start)==False:
            r_avg_start = int(np.mean(rs_avg))
            r_avg_end = int(np.mean(re_avg))
            l_avg_start = int(np.mean(ls_avg))
            l_avg_end = int(np.mean(le_avg))
            cv2.line(frame, (l_avg_start, height_max), (l_avg_end, height_min), (0, 0, 255), 5) 
            cv2.line(frame, (r_avg_start, height_max), (r_avg_end, height_min), (0, 0, 255), 5)
            pts = [(l_avg_start, height_max), (r_avg_start, height_max),(r_avg_end, height_min),(l_avg_end, height_min)]
            cv2.fillPoly(frame, np.int_([pts]), (255,255,51))
            cv2.addWeighted(frame, alpha, frame1, 1 - alpha, 0, frame1)

        else:
            cv2.line(frame, (l_start, height_max), (l_end, height_min), (0, 0, 255), 5) 
            cv2.line(frame, (r_start, height_max), (r_end, height_min), (0, 0, 255), 5)
            pts = [(l_start, height_max), (r_start, height_max),(r_end, height_min),(l_end, height_min)]
            cv2.fillPoly(frame, np.int_([pts]), (255,255,51))
            cv2.addWeighted(frame, alpha, frame1, 1 - alpha, 0, frame1)

        

        cv2.imshow('red',frame1)
        out.write(frame1)
        
      
        # Press Q on keyboard to  exit
        if cv2.waitKey(20) & 0xFF == ord('q'):
          break
                 
  else: 
    break

#Release video capture object
cap.release()
cv2.destroyAllWindows()

