import cv2
import numpy as np
import copy

cap = cv2.VideoCapture('project_video.mp4')
straight = cv2.imread('straight.jpg')

#Camera Matrix
K = np.array([[  1.15422732e+03   ,0.00000000e+00 ,  6.71627794e+02],[  0.00000000e+00 ,  1.14818221e+03 ,  3.86046312e+02],[  0.00000000e+00,   0.00000000e+00,   1.00000000e+00]])

#Distortion Coefficients
dist = np.array([[ -2.42565104e-01,  -4.77893070e-02,  -1.31388084e-03 , -8.79107779e-05,2.20573263e-02]])
font = cv2.FONT_HERSHEY_TRIPLEX
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
# Check if camera opened successfully
while(cap.isOpened()):
    
  # Capture frame-by-frame
  ret, image = cap.read()

  
  if ret == True:
        distort=cv2.undistort(image,K,dist,None,K)
        
        width = int(image.shape[1])
        height = int(image.shape[0])
        
        height_max = height-10
        height_min = int(height/1.5)

        mask = np.zeros(distort.shape, dtype = "uint8")
        cv2.rectangle(mask, (0, 425), (1500, 740), (255, 255, 255), -1)
        maskedImg = cv2.bitwise_and(distort, mask)
        hsl = cv2.cvtColor(maskedImg, cv2.COLOR_BGR2HLS)


        low_yellow = np.array([15,38,115])
        upper_yellow = np.array([35,204,255])
        mask_yellow = cv2.inRange(hsl,low_yellow,upper_yellow)


        low_white = np.array([0,200,0])
        upper_white = np.array([180,255,255])
        
#        low_white = np.array([0,150,0])
#        upper_white = np.array([180,255,255])
        mask_white = cv2.inRange(hsl,low_white,upper_white)

        hsl_mask = cv2.bitwise_or(mask_yellow,mask_white)
        hsl_combined = cv2.bitwise_and(image, image, mask = hsl_mask)


        gray=cv2.cvtColor(distort,cv2.COLOR_BGR2GRAY)

        (bottom_px, right_px) = (image.shape[0] - 1, image.shape[1] - 1) 
        src = np.array([[240,bottom_px],[595,450],[690,450], [1150, bottom_px]],np.float32)
        dst = np.array([[200,730],[200,0],[500,0],[500,730]],np.float32) 

        M = cv2.getPerspectiveTransform(src, dst)
        img_size = (image.shape[1], image.shape[0])
        #warped = cv2.warpPerspective(image, M, img_size)
        warped = cv2.warpPerspective(hsl_combined, M, img_size)
        warped1=copy.deepcopy(image)
        gray = cv2.cvtColor(warped,cv2.COLOR_BGR2GRAY)
        blurr = cv2.GaussianBlur(gray, (5, 5), 0)
        blurr=cv2.medianBlur(gray,5)
        edges = cv2.Canny(blurr,50,100)
        
        #cv2.imshow("a",blurr)
#
        histogram = np.sum(gray[gray.shape[0]//2:,:], axis=0)
        histo=histogram[0:650]
        mp=np.int(histo.shape[0]/2) 
        
        x_range = np.argwhere(histo>5000)
        index = np.where(x_range<mp)
        ii = np.shape(index)[1]
        xr_range = x_range[ii:len(x_range)]
        xl_range = x_range[0:ii]
        xr_max = max(xr_range)+50  
        xr_min = min(xr_range)-50
        xl_max = max(xl_range)
        xl_min = min(xl_range) 
        #print(xr_min)
        #print(xr_max)
               
        indices = np.argwhere(gray > 170)
        x = indices[:,1]
        y = indices[:,0]
        
        x_right = []
        y_right = []
        x_left = []
        y_left = []

        
        for i in range(0,len(x)):
            if x[i]<xl_max[0] and x[i]>xl_min[0]:
  #          if x[i]<(mp):
                
                x_left.append(x[i])
                y_left.append(y[i])
        #        cv2.circle(warped,(x[i],y[i]), 3,(0,255,0),-1)            
                
            if x[i]>xr_min[0] and x[i]<xr_max[0]:
       #     if x[i]>(mp):
                
                x_right.append(x[i])
                y_right.append(y[i])
               # cv2.circle(warped,(x[i],y[i]), 3,(0,255,0),-1)

        
        leftLine_curve = np.poly1d(np.polyfit(y_left, x_left,2))
        rightLine_curve = np.poly1d(np.polyfit(y_right, x_right, deg = 2))
       

        
        y_lc = np.linspace(720,150, num=100)
        x_lc = leftLine_curve(y_lc)
        pts_l = np.vstack((x_lc,y_lc)).astype(np.int32).T
        cv2.polylines(warped,  [pts_l],  False,  (0, 0, 255),  10)
        
        y_rc = np.linspace(720,150, num=100)
        x_rc = rightLine_curve(y_rc)
        pts_r = np.vstack((x_rc,y_rc)).astype(np.int32).T
        cv2.polylines(warped,  [pts_r],  False,  (0, 0, 255),  10)
        
        xl_start = int(x_lc[0])
        yl_start = height_max
        xl_end = int(x_lc[len(x_lc)-1])
        yl_end = int(y_lc[len(y_lc)-1])
        
        
        xr_start = int(x_rc[0])
        yr_start = height_max
        
        avgxr =int(np.mean(x_rc))
        avgyr = int(np.mean(y_rc))
        
        xr_end = int(x_rc[len(x_rc)-1])
        yr_end = int(y_rc[len(y_rc)-1])
        
        avgxl = int(np.mean(x_lc))
        avgyl = int(np.mean(y_lc))
        
        frame1=copy.deepcopy(image)
        warped1=copy.deepcopy(warped)
        pts2 = np.vstack([(xr_start, yr_start),(xl_start,yl_start),(avgxl,avgyl),(xl_end,yl_end),(xr_end, yr_end),(avgxr,avgyr)])
        cv2.fillPoly(warped1,[pts2],(255,191,0))
        warped = cv2.addWeighted(warped, 1, warped1, 0.2, 0)
        M_inv = cv2.getPerspectiveTransform(dst,src)
        frame = cv2.warpPerspective(warped, M_inv, img_size)
        result = cv2.addWeighted(frame1, 1, frame, 1, 0)

        m_left = (y_lc[len(y_lc)-1]-y_lc[0])/(x_lc[len(x_lc)-1]-x_lc[0])
        m_right = (y_rc[len(y_rc)-1]-y_rc[0])/(x_rc[len(x_rc)-1]-x_rc[0])       
           
        M_inv = cv2.getPerspectiveTransform(dst,src)
        frame = cv2.warpPerspective(warped, M_inv, img_size)
        result = cv2.addWeighted(image, 0.9, frame, 1, 0)
        
        print(m_left)
    
        if m_left<25 and m_left>0:
            cv2.putText(result,'Turning Left',(500,300), font, 1.5,(255,255,255),1,cv2.LINE_AA)        
        elif m_left>-25 and m_left<0:
            cv2.putText(result,'Turning Right',(500,300), font, 1.5,(255,255,255),1,cv2.LINE_AA)
        else:
            cv2.putText(result,'Going Straight',(500,300), font, 1.5,(255,255,255),1,cv2.LINE_AA)
        out.write(result)
        warped=cv2.resize(warped,(0,0),fx=0.7,fy=0.7)
        cv2.imshow('Warped', warped)
        result=cv2.resize(result,(0,0),fx=0.7,fy=0.7)
        cv2.imshow('Resulting Video', result)
        
        # Press Q on keyboard to  exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
          break
  #break                   
  else: 
    break

#Release video capture object
cap.release()
cv2.destroyAllWindows()