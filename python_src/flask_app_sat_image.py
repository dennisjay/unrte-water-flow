import sys

sys.path.insert( 0, './spacenet_three/src')

from io import BytesIO

from flask import Flask, send_file
from urllib import request
from PIL import Image
from itertools import chain

import torch
import numpy as np
import re

from LinkNet import LinkNet34
from presets import preset_dict
import torchvision.transforms as transforms

BING_API_KEY = 'Audnt4rCjB-fgd659_se2h2OriFlSWvuPOfzbfAspbt7QghsKj_XkJK1g_OJKQPm'
BASEURL = "http://h0.ortho.tiles.virtualearth.net/tiles/a{0}.jpeg?g=131"


checkpoint = torch.load('./norm_ln34_mul_ps_vegetation_aug_dice_best.pth')

model = LinkNet34(num_channels=3, num_classes=1)
model = torch.nn.DataParallel(model).cuda()

model.load_state_dict(checkpoint['state_dict'])


imsize = 256
loader = transforms.Compose([transforms.Scale(imsize), transforms.ToTensor()])

def image_loader(image):
    """load image, returns cuda tensor"""
    image = loader(image).float()
    image = torch.autograd.Variable(image, requires_grad=True)
    image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    return image.cuda()  #assumes that you're using GPU


app = Flask(__name__)

print( "APP READY")

def tileXY_to_quadkey(tileX, tileY, level):
    """Converts tile XY coordinates into a QuadKey at a specified level of detail
    interleaving tileY with tileX

    Arguments:
        tileX {[int]} -- [Tile X coordinate]
        tileY {[int]} -- [Tile Y coordinate]
        level {[int]} -- [Level of detail, from 1 (lowest detail) to 23 (highest detail)]

    Returns:
        [string] -- [A string containing the QuadKey]
    """

    tileXbits = '{0:0{1}b}'.format(tileX, level)
    tileYbits = '{0:0{1}b}'.format(tileY, level)

    quadkeybinary = ''.join(chain(*zip(tileYbits, tileXbits)))
    return ''.join([str(int(num, 2)) for num in re.findall('..?', quadkeybinary)])
    # return ''.join(i for j in zip(tileYbits, tileXbits) for i in j)

def serve_pil_image(pil_img):
    img_io = BytesIO()
    pil_img.save(img_io, 'JPEG', quality=70)
    img_io.seek(0)
    return send_file(img_io, mimetype='image/jpeg')


def do_inference(raw_img):
    print(model)

    t = model(image_loader(raw_img))

    print(t)



    return Image.fromarray(t.cpu().detach().numpy().reshape(256, 256) > 0.5, mode='1')




# We can use url_for('foo_view') for reverse-lookups in templates or view functions
@app.route('/raw/<z>/<x>/<y>.png')
def raw(z, x, y):
    qk = tileXY_to_quadkey(int(x), int(y), int(z))
    url = BASEURL.format(qk)
    print(url)


    with request.urlopen(url) as file:
        raw_img = Image.open(file)
        raw_img.save('raw.jpeg')

        pred_img = do_inference(raw_img)



        return serve_pil_image(pred_img)




if __name__ == '__main__':
    do_inference(Image.open('raw.jpeg'))
    app.run()

