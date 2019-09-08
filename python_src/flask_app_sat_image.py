import sys

import albumentations
from albumentations.pytorch.functional import img_to_tensor

from xd_xd import unet_vgg16

sys.path.insert(0, './spacenet_three/src')

from io import BytesIO

from flask import Flask, send_file
from urllib import request
from PIL import Image
from itertools import chain

import torch
import numpy as np
import re

BING_API_KEY = 'Audnt4rCjB-fgd659_se2h2OriFlSWvuPOfzbfAspbt7QghsKj_XkJK1g_OJKQPm'
BASEURL = "http://h0.ortho.tiles.virtualearth.net/tiles/a{0}.jpeg?g=131"
OSM_TILES = 'https://a.tile.openstreetmap.de/{2}/{0}/{1}.png'


checkpoint = torch.load('xdxd_spacenet4_solaris_weights.pth', map_location={'cuda:0': 'cpu'})
model = unet_vgg16(pretrained=False)

print(checkpoint)

if 'module.final.weight' in checkpoint:
    model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(checkpoint)
    model = model.module
else:
    model.load_state_dict(checkpoint)

loader = albumentations.Compose([albumentations.CenterCrop(256, 256, p=1.0), albumentations.Normalize()])


def image_loader(image):
    image = np.asarray(image)
    """load image, returns cuda tensor"""
    image = loader(image=image)['image']
    image = img_to_tensor(image)
    image = torch.autograd.Variable(image, requires_grad=True)

    return image.unsqueeze(0)


app = Flask(__name__)

print("APP READY")


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
    pil_img.save(img_io, 'PNG')
    img_io.seek(0)
    return send_file(img_io, mimetype='image/png')


def do_inference(raw_img):
    raw_img.save('raw.jpeg')

    outputs = model(image_loader(raw_img))
    y_pred_sigmoid = np.clip(torch.sigmoid(
        torch.squeeze(outputs)
    ).detach().cpu().numpy(), 0.0, 1.0)

    bit_mask = (y_pred_sigmoid > 0.005)
    return bit_mask


# We can use url_for('foo_view') for reverse-lookups in templates or view functions
@app.route('/raw/<z>/<x>/<y>.png')
def raw(z, x, y):
    qk = tileXY_to_quadkey(int(x), int(y), int(z))
    url = BASEURL.format(qk)
    osm_url = OSM_TILES.format(x, y, z)

    print(osm_url, url)


    with request.urlopen(url) as file:
        with request.urlopen(osm_url) as file2:
            raw_img = Image.open(file)
            raw_img_np = np.uint8(np.asarray(raw_img)).copy()
            osm_map_np = np.uint8(np.asarray(Image.open(file2).convert('RGB'))).copy()

            bit_mask = do_inference(raw_img)

            raw_img_np[~bit_mask] = 0
            osm_map_np[bit_mask] = 0

            img = Image.fromarray(raw_img_np + osm_map_np, 'RGB')



            return serve_pil_image(img)


if __name__ == '__main__':
    app.run()

