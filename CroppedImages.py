# Create cropped images from face images 

from PIL import Image
from os import listdir


def get_concat_v_cut(im1, im2):
    dst = Image.new('RGB', (min(im1.width, im2.width), im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst


def crop_image(image_path):
  im = Image.open(image_path)
  im = im.resize((196, 235))
  imeyes = im.crop((30,72,180,121))
  immouth = im.crop((30,162,180,232))
  pic = get_concat_v_cut(imeyes,immouth)
  return (True,pic)




count = 0
for i in listdir('./real'):  #Image path 
    if (count >= 0):
        print(i)
        img_path = './real/' + i
        filename = str(count) + '.jpg'
        status,img=  crop_image(img_path)
        if status :
            name = "." + str(count) + ".jpg"
            img.save(name,"JPEG")
       
    count += 1
