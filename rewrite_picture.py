
import cv2
from PIL import ImageFont, ImageDraw, Image
import numpy as np
import PIL
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import os
import shutil
import numpy as np

def addTextTOImage(imageSrc, newImage):
    #
    image = Image.open(imageSrc)
    draw = ImageDraw.Draw(image)
    textsize = 20
    ft = ImageFont.truetype("/home/z840/poweryin/arialuni.ttf", textsize)
    # font = ImageFont.truetype("C:\\WINDOWS\\Fonts\\SIMYOU.TTF", 20)
    #draw.text((5, 5), "Anomaly!!!", (255), font=font)
    draw.text((150,200), "score",font=ft,fill=(0,0,0))
    draw.text((800,900), "frame number",font=ft,fill=(0,0,0))
    ImageDraw.Draw(image)
    # 保存图片
    image.save(newImage)

imagepath="/data/UCF_Crimes/c3d_features/curve_t/"
add_path="/data/UCF_Crimes/c3d_features/correct/"
image = os.listdir(imagepath)
# image.sort(key=lambda i: int(re.match(r'(\d+)', i).group()))
for tmp in image:

    addTextTOImage(os.path.join(imagepath,tmp),os.path.join(add_path,tmp))
