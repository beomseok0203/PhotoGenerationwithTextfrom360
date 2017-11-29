import skimage.data
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import selectivesearch
from PIL import Image
import numpy
def main():
    file_path = './splited_image/'
    for i in range(8):
        print 'object proposal box detecting... %d/8' %(i+1)

        image_name = 'test'+str(i)+'.jpg'
        full_name = file_path +image_name
        txt_full_name ='./proposal_box/selective_box'+str(i) +'.txt'
        #f =open("selective_box2.txt","w")
        f =open(txt_full_name,"w")

        # loading astronaut image
        #img = numpy.asarray(Image.open('./test_image/test1.jpg'))
        img = numpy.asarray(Image.open(full_name))
        #img = skimage.data.astronaut()

        # perform selective search
        img_lbl, regions = selectivesearch.selective_search(
            img, scale=500, sigma=0.9, min_size=10)

        candidates = set()
        for r in regions:
            # excluding same rectangle (with different segments)
            if r['rect'] in candidates:
                continue
            # excluding regions smaller than 2000 pixels
            if r['size'] < 2000:
                continue
            # distorted rects
            x, y, w, h = r['rect']
            if w / h > 2.5 or h / w > 2.5:
                continue
            candidates.add(r['rect'])
            #print r['rect']

        print 'candidates box # : %d' % len(candidates)
        #print len(candidates)
        #fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
        #ax.imshow(img)
        for x, y, w, h in candidates:
            #print x, y, w, h
            data = "%d\t"%x
            f.write(data)

            data = "%d\t" % y
            f.write(data)

            data = "%d\t" % (x+w-1)
            f.write(data)

            data = "%d\t\n" % (y+h-1)
            f.write(data)


            rect = mpatches.Rectangle(
                (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
            #ax.add_patch(rect)

        #plt.show()
        f.close()
if __name__ == "__main__":
    main()
