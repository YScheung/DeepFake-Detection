from PIL import Image
from PIL import Image


im = Image.open("./face/7.jpg")
im = im.resize((196, 235))
imeyes = im.crop((30,70,180,112))
#imeyes.show()
immouth = im.crop((30,145,180,232))
#immouth.show()



def get_concat_v_cut(im1, im2):
    dst = Image.new('RGB', (min(im1.width, im2.width), im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

aa = get_concat_v_cut(imeyes,immouth)
aa.show()

