#from __future__ import division
import math
from PIL import Image
import numpy as np
import datetime


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
def make_2Dimage(img,u,v,fovX,fovY,lookup_index):
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

            #dir_0 = xx*basis[0][0,0] + yy*basis[1][0,0] - basis[2][0,0]
            #dir_1 = xx * (basis[0])[1,0] + yy * basis[1][1,0] - basis[2][1,0]
            #dir_2 = xx * basis[0][2,0] + yy * basis[1][2,0] - basis[2][2,0]
            dir_0 = xx*x0 + yy*y0 - z0
            dir_1 = xx * x1 + yy *y1 - z1
            dir_2 = xx * x2+ yy * y2 - z2
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
            tuple1 = (lookup_index,x,y)
            tuple2 = (ix,iy)
            lookup[tuple1] = tuple2

    return (img2,lookup)

        #break
    #break
img_index=0
image_path ='./splited_image/test'

img = Image.open('./test_panorama/pano_aohmbsejqkhebw.jpg')
img = img.convert('RGB')
pixel = img.load()


        #I tries as mentionned in the following code to invert x and y in the two previous lines but the index error out of range comes back
##img2.show()


f1 = open("./2D_index.txt",'r')
lines1 = [line1.rstrip() for line1 in f1.readlines()]
for line1 in lines1:

    print 'making cubemap image... %d/8' % (img_index + 1)
    index = line1.split('\t')[0]
    u = line1.split('\t')[1]
    v = line1.split('\t')[2]
    a = datetime.datetime.now()
    tmp = make_2Dimage(img, float(u), float(v), math.pi / 1.8, math.pi / 1.8,img_index)
    print (datetime.datetime.now() - a)
    img2 = tmp[0]
    table = tmp[1]
    #print table
    image_name = image_path + str(index) + '.jpg'
    img2.save(image_name)
    img_index =img_index +1

    if img_index>7:
        break

    #print "{},{}".format(u,v)
    #dir_key = dir_key.replace("\\", "/")