from PIL import Image
import os
size = 740,360


for i in range(0,10):
    count = 0
    path = "/home/anshuman/mnist_png/testing/"+str(i)
    for filename in os.listdir(path):
        #filename = os.fsdecode(file)
        print filename
        if filename.endswith(".png") :
            count+=1
            new_path = path+'/'+filename
            print new_path
            im = Image.open(new_path)
            im_resized = im.resize(size,Image.ANTIALIAS)
            save_path = "/home/anshuman/mnist_png/new_testing/"+str(i)+'/'+filename
            im_resized.save(save_path,"PNG")
        if count>9 :
            break
'''
im = Image.open("profilepic.jpg")
im_resized = im.resize(size, Image.ANTIALIAS)
im_resized.save("my_image_resized.jpg", "JPEG")
'''
