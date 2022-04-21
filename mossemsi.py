# Program To Read video
# and Extract Frames
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from skimage.filters import median
from skimage.morphology import disk
from matplotlib.patches import Rectangle

def show_images(images: list, titles: list="Untitled    ", colorScale='gray', rows = 0, columns = 0) -> None:
    n: int = len(images)
    if rows == 0:
        rows=int(math.sqrt(n))
    if columns == 0:
        columns=(n/rows)
    f = plt.figure()
    for i in range(n):
        # Debug, plot figure
        f.add_subplot(rows, columns, i + 1)
        plt.imshow(images[i], cmap=colorScale)
        plt.title(titles[i])
    plt.show(block=True)

def show_rect(image,rect):
    plt.imshow(image)
    plt.gca().add_patch(rect)
    plt.show()
    
def get_rect(centroid,size):
    x1 = centroid[0]-size[0]//2
    x2 = centroid[0]+size[0]//2
    y1 = centroid[1]-size[1]//2
    y2 = centroid[1]+size[1]//2
    return (x1,y1),(x2,y2)  

def crop_rect(image,centroid,size):
    x1 = centroid[0]-size[0]//2
    x2 = centroid[0]+size[0]//2
    y1 = centroid[1]-size[1]//2
    y2 = centroid[1]+size[1]//2
    roi = image[y1:y2, x1:x2]
    ref = (x1,y1)
    return roi, ref

def get_centroid(x1,y1,x2,y2):
    centroid = ((x1+x2)//2,(y1+y2)//2)
    size = (x2-x1,y2-y1)
    return centroid,size
  
def match_filter(image,my_filter):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_filter = cv2.cvtColor(my_filter, cv2.COLOR_BGR2GRAY)
    result = cv2.matchTemplate(image, my_filter, cv2.TM_CCOEFF_NORMED)
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(result)
    (x1, y1) = maxLoc
    x2 = x1 + my_filter.shape[1]
    y2 = y1 + my_filter.shape[0]
    #print(maxVal)
    return (x1,y1),(x2,y2), maxVal
    
def match_filters(image,my_filter):
    (x1,y1),(x2,y2), maxVal = match_filter(image,my_filter)
    scale_percent = 97 # percent of original size
    width = int(my_filter.shape[1] * scale_percent / 100)
    height = int(my_filter.shape[0] * scale_percent / 100)
    dim = (width, height)
    small_filter = cv2.resize(my_filter, dim, interpolation = cv2.INTER_AREA)
    (x1s,y1s),(x2s,y2s), maxVals = match_filter(image,small_filter)
    #print(maxVals,maxVal)
    if(maxVals<maxVal):
        return (x1,y1),(x2,y2), maxVal
    else:
        print("go small")
        return (x1s,y1s),(x2s,y2s), maxVals

def update_filter(old_filter,new_filter):
    if(old_filter.shape[1]!=new_filter.shape[1]):
        width = new_filter.shape[1]
        height = new_filter.shape[0]
        dim = (width, height)
        old_filter = cv2.resize(old_filter, dim, interpolation = cv2.INTER_AREA)
    updated_filter = new_filter//16 + (15*old_filter)//2
    return updated_filter
        
def predict_trace(positions):
    print(positions[-1][0])
    try:
        px = [positions[-2][0],positions[-1][0]]
        py = [positions[-2][1],positions[-1][1]]
        t = [-2,-1]
        xcoefficients = np.polyfit(t, px, 1)
        ycoefficients = np.polyfit(t, py, 1)
        xpolynomial = np.poly1d(xcoefficients)
        ypolynomial = np.poly1d(ycoefficients)
        newx = xpolynomial(0)
        newy = ypolynomial(0)
        print ('newx =', newx)
        print ('newy =', newy)
        return int(x), int(y)
    except:
        return positions[-1][0], positions[-1][1]
    
def reference_correction(x1,y1,x2,y2,ref):
    x1 = ref[0] + x1
    x2 = ref[0] + x2
    y1 = ref[1] + y1
    y2 = ref[1] + y2
    return (x1,y1),(x2,y2)
    
def FrameCapture(path, time, centroid, size):
    
    # Path to video file
    vidObj = cv2.VideoCapture(path)

    # Used as counter variable
    count = 0

    # checks whether frames were extracted
    success = 1

    th = 2
    color = (255,0,0)
    positions = []
    roi_c = 2
    while success:

        # vidObj object calls read
        # function extract frames
        success, image = vidObj.read()
        
        if(count==time):
            (x1,y1),(x2,y2) = get_rect(centroid,size)
            image = cv2.rectangle(image, (x1-th,y1-th), (x2+th,y2+th), color, th)
            my_filter,ref = crop_rect(image,centroid,size)
            new_filter = my_filter
            pred_cent = centroid
            roi_size = (roi_c*size[0],roi_c*size[1])
            new_roi,ref = crop_rect(image,centroid,roi_size)

            
            show_images([image,my_filter])
        
        
        if (count>time):
            new_roi,ref = crop_rect(image,pred_cent,roi_size)
            print(ref)
            #scan image to match filter
            window_name1 = 'roi'
            cv2.imshow(window_name1, new_roi)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
            (x1,y1),(x2,y2), maxVal = match_filter(new_roi,new_filter)
            print((x1,y1),(x2,y2))
            (x1,y1),(x2,y2) = reference_correction(x1,y1,x2,y2,ref)
            print((x1,y1),(x2,y2))
            print("NEXT: ******************************")
            image = cv2.rectangle(image, (x1-th,y1-th), (x2+th,y2+th), color, th)
            old_filter = new_filter
            new_filter = image[y1:y2, x1:x2,:]
            updated_filter = update_filter(old_filter,new_filter)
            #show_images([image,my_filter,old_filter,new_filter])
            centroid,size = get_centroid(x1,y1,x2,y2)
            positions.append(centroid)
            pred_cent = predict_trace(positions)
            roi_size = (roi_c*size[0],roi_c*size[1])

            
            window_name = 'tracking'
            cv2.imshow(window_name, image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
        count += 1
# Driver Code
if __name__ == '__main__':

    # path = "C:\\Users\\User\\Downloads\\Motocross - 10797.mp4"
    # time = 23
    # centroid = (295,480)
    # size = (50,80)
    
    # path = "C:\\Users\\User\\Downloads\\Helicopter - 40672.mp4"
    # time = 1
    # centroid = (200,140)
    # size = (75,40)    
    
    path = "C:\\Users\\User\\Downloads\\Plane - 3966.mp4"
    time = 100
    centroid = (270,240)
    size = (120,40)  
    FrameCapture(path, time, centroid, size)
