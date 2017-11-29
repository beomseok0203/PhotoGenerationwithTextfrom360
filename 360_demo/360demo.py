from __future__ import division, print_function
from PIL import Image
import sys
import matplotlib
import numpy as np
import skimage.io
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
#matplotlib inline
sys.path.append('../')
#sys.path.append('../external/caffe-natural-language-object-retrieval/')
sys.path.insert(0,'../external/caffe-natural-language-object-retrieval/python/')
sys.path.append('../external/caffe-natural-language-object-retrieval/examples/coco_caption/')
import caffe
import scipy.misc
import numpy as np
import util
from captioner import Captioner
import retriever
import math
#import image_convert

#im_file = './demo_data/test2.jpg'



pretrained_weights_path = '../models/two_layer_LSTM.caffemodel'
gpu_id = 0

# Initialize the retrieval model
image_net_proto = '../prototxt/VGG_ILSVRC_16_layers_deploy.prototxt'
lstm_net_proto = '../prototxt/scrc_word_to_preds_full.prototxt'
vocab_file = '../data/vocabulary.txt'
# utilize the captioner module from LRCN
captioner = Captioner(pretrained_weights_path, image_net_proto, lstm_net_proto,
                      vocab_file, gpu_id)
captioner.set_image_batch_size(50)  # decrease the number if your GPU memory is small
vocab_dict = retriever.build_vocab_dict_from_captioner(captioner)

while(1):
    sum_candidate_box =[]
    sum_score_box =[]

    query =raw_input("type the input query: ")
    #query = 'bike on the red house'

    print ("query =",query)
    print("Find best candidate..!")
    for i in range(8):
        im_file = './splited_image/test'+str(i)+'.jpg'
        edgebox_file = './proposal_box/selective_box'+str(i)+'.txt'  # pre-extracted EdgeBox proposals
        im = skimage.io.imread(im_file)
        imsize = np.array([im.shape[1], im.shape[0]])  # [width, height]
        candidate_boxes = np.loadtxt(edgebox_file).astype(int)
        candidate_boxes =np.reshape(candidate_boxes,(-1,4))
        # Compute features
        region_feature = retriever.compute_descriptors_edgebox(captioner, im,
                                                            candidate_boxes)
        spatial_feature = retriever.compute_spatial_feat(candidate_boxes, imsize)
        descriptors = np.concatenate((region_feature, spatial_feature), axis=1)
        context_feature = captioner.compute_descriptors([im], output_name='fc7')

        # Compute scores of each candidate region
        scores = retriever.score_descriptors_context(descriptors, query,
                                                     context_feature, captioner,
                                                     vocab_dict)
        #candidate_boxes = (i, candidate_boxes)
        candidate_boxes=np.insert(candidate_boxes,0,i,axis=1 )
        if(i==0):
            sum_candidate_box=candidate_boxes;
        else:
            #sum_candidate_box=np.concatenate(sum_candidate_box,candidate_boxes,axis=1)
            sum_candidate_box = np.vstack((sum_candidate_box,candidate_boxes))
        sum_score_box=np.concatenate((sum_score_box, scores))

    #print (sum_score_box)
    #print (sum_candidate_box)
    # Retrieve the top-scoring candidate region given the query
    #retrieved_bbox = candidate_boxes[np.argmax(scores)]
    retrieved_bbox = sum_candidate_box[np.argmax(sum_score_box)]
    # Visualize the retrieval result
    #plt.figure(figsize=(12, 8))
    #plt.imshow(im)


    #im.save('tmp.jpg')
    #ax = plt.gca()
    index,x_min, y_min, x_max, y_max = retrieved_bbox
    #ax.add_patch(mpatches.Rectangle((x_min, y_min), x_max-x_min+1, y_max-y_min+1,
    #                                fill=False, edgecolor='r', linewidth=5))
    #_ = plt.title("query = '%s'" % query)
    scipy.misc.imsave('outfile.jpg', im)
    crop_im_file = './splited_image/test'+str(index)+'.jpg'
    img_forcrop = Image.open(crop_im_file)
    cropped = img_forcrop.crop((x_min,y_min,x_max,y_max))
    cropped.save("cropped.jpg")
    center_x =int( (x_min+x_max)/2)
    center_y =int( (y_min+y_max)/2)
    key=(center_x,center_y)


    def makebasis(u,v):
        yaw = (0.5 - u) * (-2.0*math.pi);
        pitch = (0.5 - v) * (math.pi);
        R_p =np.matrix(([1.0, 0.0, 0.0],[ 0.0 ,math.cos(pitch),math.sin(pitch)],[0.0,-math.sin(pitch),math.cos(pitch)]))
        R_y =np.matrix(([math.cos(yaw),0.0 , -math.sin(yaw)],[0.0,1.0,0.0],[math.sin(yaw),0.0,math.cos(yaw)]))
        R_p = R_p.getT()
        R_ = np.matmul(R_y,R_p)

        x_basis = R_[:, 0]
        y_basis = R_[:, 1]
        z_basis = R_[:, 2]
        return [x_basis,y_basis,z_basis]
    #img.show()
    def make_2Dimage(img,u,v,fovX,fovY):
        inwidth, inheight = img.size

       # fovX =math.pi/2
        #fovY =math.pi/2
        outwidth =700
        #outheight = int(outwidth * fovY/fovX)
        outheight=700

        img2 = Image.new(img.mode,(outwidth,outheight),0)
        #generate float image code
        basis =makebasis(u,v) #center point (u,v) (0,0-left top)
        x0 = basis[0][0, 0]
        x1 = basis[0][1, 0]
        x2 = basis[0][2, 0]
        y0 = basis[1][0, 0]
        y1 = basis[1][1, 0]
        y2 = basis[1][2, 0]
        z0 = basis[2][0, 0]
        z1 = basis[2][1, 0]
        z2 = basis[2][2, 0]
        #print basis
        lookup = dict();

        for y in xrange(outheight):
            #print y
            for x in xrange(outwidth):
                #xx = 2*(y+0.5) / width - 1.0
                xx = ((x-outwidth/2.0)/outwidth * math.tan(fovX/2.0) *2.0)
                #yy = 2*(y+0.5)/ height - 1.0
                yy = ((outheight/2.0 -y) /outheight *math.tan(fovY/2.0)*2.0)


                dir_0 = xx * x0 + yy * y0 - z0
                dir_1 = xx * x1 + yy * y1 - z1
                dir_2 = xx * x2 + yy * y2 - z2
               # print outheight
                #print 'index xx %f' %xx
                #print 'math tan %f' %math.tan(fovX/2.0)

                len = math.sqrt(dir_0*dir_0 +dir_1*dir_1+ dir_2*dir_2)
                #ix and iy must be integers
                #ix = int((0.5 * lng / math.pi + 0.5) * width - 0.5)
                ix = int((math.atan2( dir_0,-(dir_2))/(2*math.pi)+ 0.5)*inwidth)
                iy = int((0.5-math.asin( dir_1/len)/math.pi)*inheight )
                if(ix>=inwidth):
                    ix= inwidth-1
                if(ix<0):
                    ix=0
                if(iy>=inheight):
                    iy = inheight -1
                if(iy<0):
                    iy=0




                #not sure of this part to remap the image
                newpixel = pixel[ix, iy]
                #print xx
                img2.putpixel([x, y], newpixel)
                tuple1 = (x,y)
                tuple2 = (ix,iy)
                lookup[tuple1] = tuple2

        return (img2,lookup)

    ####find equirectangular coordinate

    u=0
    v=0
    img = Image.open('./test_panorama/pano_aohmbsejqkhebw.jpg')
    img = img.convert('RGB')
    im_size = img.size
    pixel = img.load()
    f1 = open("./2D_index.txt",'r')
    lines1 = [line1.rstrip() for line1 in f1.readlines()]
    for line1 in lines1:
        index_str = line1.split('\t')[0]
        if(int(index_str) ==index ):
            u = line1.split('\t')[1]
            v = line1.split('\t')[2]
            print("making 2D good view..")
            tmp = make_2Dimage(img, float(u), float(v), math.pi / 1.8, math.pi / 1.8)
            table = tmp[1]

    value =table.get(key)

    ####making 2D view image

    tmp_ = make_2Dimage(img,value[0]/im_size[0],value[1]/im_size[1],math.pi*59.5/180,math.pi*46.3/180)
    if(abs(x_max-x_min)>450|(y_max-y_min)>450):
        tmp_ = make_2Dimage(img, value[0] / im_size[0], value[1] / im_size[1], math.pi * 100/ 180, math.pi * 90 / 180)
    re_img = tmp_[0].resize((700,525),Image.BILINEAR)
    scipy.misc.imsave('outfile1.jpg', re_img)


    #fig = plt.figure()

    #a = fig.add_subplot(1, 2, 1)
    #imgplot = plt.imshow(cropped)
    #a.set_title('Cropped Region')
    #im = plt.imshow(cropped)
    #a = fig.add_subplot(1, 2, 2)
    #plt.show()
    #plt.close()
    #_ = plt.title("2D good view")
    #im = plt.imshow(re_img)
    #a.set_title('2D image')
    #imgplot = plt.imshow(tmp_[0])
    #plt.show()
    #plt.close()
