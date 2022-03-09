
from collections import defaultdict
from fileinput import filename

from this import d
from types import DynamicClassAttribute
from flask import Flask,request,jsonify,render_template, send_file
from pyparsing import Word
#from skimage import io
from PIL import Image
import os
import skimage.filters

import cv2 as cv
import numpy as np

import werkzeug

app = Flask(__name__)
  

@app.route('/upload',methods = ['POST'])        
def upload():
    if(request.method == 'POST'):
        imagefile = request.files['image']
        global filename 
        filename = werkzeug.utils.secure_filename(imagefile.filename)
        imagefile.save("./uploadedimages/"+filename) 
        return jsonify({
            "message":"Image Uploaded Successfully"
        })


@app.route('/app',methods = ['GET'])  
def returnascii():
    d = {};
    inputchar = str(request.args['query'])
    answer = str(ord(inputchar))
    d['output']=answer
    # abc = "hwllo sldlfjew;o;h"
    return d

@app.route('/test',methods = ['GET'])  
def returntest():
    # d = {};
    # inputchar = str(request.args['testing'])
    # answer = str(ord(inputchar))
    # d['output']=answer
    # abc = "hwllo sldlfjew;o;h"
    affectedpercentage1= "70 percent"
    mesg1 = "Mild Stage Cataract"
    return jsonify({'affectedpercenatge':affectedpercentage1,
                            'message':mesg1
                        })


def preprocessing(img):
    if (img.shape[0] > 120 and img.shape[1] >120):
      roi_img = cv.resize(img,(120,120)) #resizing 120*120 pixels
    else:
        roi_img= img
    R, G, B = roi_img[:,:,0], roi_img[:,:,1], roi_img[:,:,2] #get the red, blue and green dimension matrices of the RGB image 
    imgGray = 0.2989 * R + 0.5870 * G + 0.1140 * B
    
    g_kernel = gkernel(5,1)
    global dst_IMG
    dst_IMG = cv.filter2D(imgGray,-1,g_kernel)
    return dst_IMG
 # """ Gaussian Kernel Creator via given length and sigma"""
def gkernel(l=5, sig=1):
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    xx, yy = np.meshgrid(ax, ax)

    kernel = (1/(2*3.14*np.square(sig)))*(np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig)))

    return kernel / np.sum(kernel)


def find_hough_circles(img, edge_image, r_min, r_max, delta_r, num_thetas, bin_threshold, post_process = True):
    img_height, img_width = edge_image.shape[:2] #image size
    dtheta = int(360 / num_thetas)# R and Theta ranges
    thetas = np.arange(0, 360, step=dtheta)
    rs = np.arange(r_min, r_max, step=delta_r)
    cos_thetas = np.cos(np.deg2rad(thetas))
    sin_thetas = np.sin(np.deg2rad(thetas))

    circle_candidates = []
    for r in rs:
        for t in range(num_thetas):
            circle_candidates.append((r, int(r * cos_thetas[t]), int(r * sin_thetas[t])))
    
    accumulator = defaultdict(int)

    for y in range(img_height):
        for x in range(img_width):
            if edge_image[y][x] != 0: #white pixel
        # Found an edge pixel so now find and vote for circle from the candidate circles passing through this pixel.
                for r, rcos_t, rsin_t in circle_candidates:
                    x_center = x - rcos_t
                    y_center = y - rsin_t
                    accumulator[(x_center, y_center, r)] += 1

    output_img = img.copy()

    out_circles = []

    for candidate_circle, votes in sorted(accumulator.items(), key=lambda i: -i[1]):
        x, y, r = candidate_circle
        current_vote_percentage = votes / num_thetas
        if current_vote_percentage >= bin_threshold: 
      # Shortlist the circle for final result
            out_circles.append((x, y, r, current_vote_percentage))

    if post_process :
        pixel_threshold = 5
        postprocess_circles = []
        for x, y, r, v in out_circles:
            if all(abs(x - xc) > pixel_threshold or abs(y - yc) > pixel_threshold or abs(r - rc) > pixel_threshold for xc, yc, rc, v in postprocess_circles):
                postprocess_circles.append((x, y, r, v))
        out_circles = postprocess_circles

    for x, y, r, v in out_circles:
        output_img = cv.circle(output_img, (x,y), r, (0,255,0), 2)

    return output_img, out_circles
  

def percentcalculate(img):
    t = skimage.filters.threshold_otsu(img)
        # create a binary mask with the threshold found by Otsu's method    
    binary_mask = img > t
        #plt.imshow(binary_mask, cmap='gray')
        #plt.show()
    affectedPixels = np.count_nonzero(binary_mask)
    totalpixels= binary_mask.size

    affectedpercenatge=(affectedPixels/totalpixels)*100
    if affectedpercenatge  > 0 and affectedpercenatge < 10:
        mesg = "MILD Stage Cataract" #print("MILD Stage Cataract")
    elif  affectedpercenatge  > 10 and affectedpercenatge < 50:
        mesg = "MODERATE stage Cataract" #print("MODERATE stage Cataract")
    elif affectedpercenatge  > 50 and affectedpercenatge < 90:
        mesg = "PRONOUNCED stage Cataract" #print("PRONOUNCED stage Cataract")
    else: 
        mesg = "SEVERE stage Cataract" #print("SEVERE stage Cataract")
    return affectedpercenatge,mesg
        
@app.route('/app1')
def result():
    try: 
        global img 
        img = Image.open(r'./uploadedimages/'+str(filename)) 
    except IOError:
        pass
    # print(img)
    print(img)
    dst_IMG = preprocessing(img);
    r_min = 10
    r_max = 200
    delta_r = 1
    num_thetas = 100
    bin_threshold = 0.4
    min_edge_threshold = 100
    max_edge_threshold = 200

    height,width = dst_IMG.shape
    mask = np.zeros((height,width), np.uint8)
       
    dst_IMG= dst_IMG.astype(np.uint8)
    edge_image = cv.Canny(dst_IMG, min_edge_threshold, max_edge_threshold)
        
    if edge_image is not None:   
        circle_img, circles = find_hough_circles(dst_IMG, edge_image, r_min, r_max, delta_r, num_thetas, bin_threshold)      
    #cv2_imshow(circle_img)
    #cv.imwrite('imageafterHOUGH1.jpeg', circle_img)
    print(circle_img)
    

    for i in circles[0::]:
        
        cv.circle(mask,(i[0],i[1]),i[2],(255,255,255),thickness=-1)

    masked_data = cv.bitwise_and(dst_IMG, dst_IMG, mask=mask)

    # Apply Threshold
    _,thresh = cv.threshold(mask,1,255,cv.THRESH_BINARY)

    # Find Contour
    contours, _ = cv.findContours(thresh,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    x,y,w,h = cv.boundingRect(contours[0])

    # Crop masked_data
    cropedimg= masked_data[y:y+h,x:x+w]


    #cv2_imshow(cropedimg) 
    #cv.imwrite('imageafterMASKING1.jpeg', cropedimg)

    """**Features extraction**"""

    # getting mean value
    mean = np.mean(cropedimg)  
    # printing mean value
    print("Mean Value for image : " ,mean)
    # getting variance
    variance = np.var(cropedimg)
    # printing varince
    print("variance for image : " ,variance)

    """Thresholding"""

    #based on texturefeatures of images, calculated the threshold value for the mean intensity and variance 
    #using diagnostic opinion-based parameter thresahold as
    mean_threshold=55.2 
    var_threshold=2200
    if mean >= mean_threshold and variance >= var_threshold:
        #cv2_imshow(roi_img)
        affectedpercenatge,mesg= percentcalculate(cropedimg)
        return jsonify({'affectedpercenatge':affectedpercenatge,
                            'message':mesg
                        })
    else:
        affectedpercenatge = 0
        mesg = "Healthy Eye"
        # print("Healthy eyes")
        return jsonify({'affectedpercenatge':affectedpercenatge,
                            'message':mesg
                        })

if __name__ == "__main__":
    app.run(debug=True)