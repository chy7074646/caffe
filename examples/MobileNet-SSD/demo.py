import numpy as np  
import sys,os  
import cv2
import time
caffe_root = '/home/ula/Ula/caffe_ssd/'
sys.path.insert(0, caffe_root + 'python')  
import caffe  


net_file= '/home/ula/Ula/caffe_ssd/examples/MobileNet-SSD/example/no_bn.prototxt'  
caffe_model='/home/ula/Ula/caffe_ssd/examples/MobileNet-SSD/example/no_bn.caffemodel'  
test_dir = "/home/ula/Ula/caffe_ssd/examples/MobileNet-SSD/images/test/"
save_detect_result="/home/ula/Ula/caffe_ssd/examples/MobileNet-SSD/images/result/"

if not os.path.exists(caffe_model):
    print(caffe_model + " does not exist")
    exit()
if not os.path.exists(net_file):
    print(net_file + " does not exist")
    exit()

caffe.set_mode_gpu()
caffe.set_device(0)
net = caffe.Net(net_file,caffe_model,caffe.TEST)  


#CLASSES = ('background',
#           'aeroplane', 'bicycle', 'bird', 'boat',
#           'bottle', 'bus', 'car', 'cat', 'chair',
#           'cow', 'diningtable', 'dog', 'horse',
#           'motorbike', 'person', 'pottedplant',
#           'sheep', 'sofa', 'train', 'tvmonitor')
CLASSES = ('background','Blob')

def preprocess(src):
    img = cv2.resize(src, (300,300))
    img = img - 127.5
    img = img * 0.007843
    return img

def postprocess(img, out):   
    h = img.shape[0]
    w = img.shape[1]
    box = out['detection_out'][0,0,:,3:7] * np.array([w, h, w, h])

    cls = out['detection_out'][0,0,:,1]
    conf = out['detection_out'][0,0,:,2]
    return (box.astype(np.int32), conf, cls)

def detect(imgfile):
    origimg = cv2.imread(imgfile)

    ##############add_by_chy_200521###################
    base_img=os.path.basename(imgfile)
    name_img=os.path.splitext(base_img)[0]
    img_save_to_path=os.path.join(save_detect_result,name_img+"_result.jpg")
    ##################################################

    img = preprocess(origimg)
    
    img = img.astype(np.float32)
    img = img.transpose((2, 0, 1))

    net.blobs['data'].data[...] = img
    out = net.forward()  
    box, conf, cls = postprocess(origimg, out)

    for i in range(len(box)):
       p1 = (box[i][0], box[i][1])
       p2 = (box[i][2], box[i][3])
       cv2.rectangle(origimg, p1, p2, (0,255,0))
       p3 = (max(p1[0], 15), max(p1[1], 15))
       title = "%s:%.2f" % (CLASSES[int(cls[i])], conf[i])
       cv2.putText(origimg, title, p3, cv2.FONT_ITALIC, 0.6, (0, 255, 0), 1)
    #cv2.imshow("SSD", origimg)
    ##############add_by_chy_200521###################
    cv2.imwrite(img_save_to_path,origimg)
    ##################################################
 
    #k = cv2.waitKey(0) & 0xff
        #Exit if ESC pressed
    #if k == 27 : return False
    return True

time_cost=0
lens=len(os.listdir(test_dir))
for f in os.listdir(test_dir):
    start=time.time()
    if detect(test_dir + "/" + f) == False:
       break
    end=time.time()
    cost=end-start
    time_cost=time_cost+cost

print("total test images:",lens,"the average time cost:",(time_cost/lens)*1000,"ms")
