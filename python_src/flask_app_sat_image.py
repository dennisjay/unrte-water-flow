from io import BytesIO

from flask import Flask, send_file
from urllib import request
from PIL import Image
from itertools import chain

import tensorflow as tf
import tensorflow_hub as hub

import numpy as np
import re


BING_API_KEY = 'Audnt4rCjB-fgd659_se2h2OriFlSWvuPOfzbfAspbt7QghsKj_XkJK1g_OJKQPm'
BASEURL = "http://h0.ortho.tiles.virtualearth.net/tiles/a{0}.jpeg?g=131"

module_handle = "https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1" #"https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"

with tf.Graph().as_default():
    detector = hub.Module(module_handle)
    image_string_placeholder = tf.placeholder(tf.string)
    decoded_image = tf.image.decode_jpeg(image_string_placeholder)
    # Module accepts as input tensors of shape [1, height, width, 3], i.e. batch
    # of size 1 and type tf.float32.
    decoded_image_float = tf.image.convert_image_dtype(
        image=decoded_image, dtype=tf.float32)
    module_input = tf.expand_dims(decoded_image_float, 0)
    result = detector(module_input, as_dict=True)
    init_ops = [tf.global_variables_initializer(), tf.tables_initializer()]

    session = tf.Session()
    session.run(init_ops)

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
    output = BytesIO()
    raw_img.save(output, format='JPEG')
    image_string = output.getvalue()

    result_out, image_out = session.run(
        [result, decoded_image],
        feed_dict={image_string_placeholder: image_string})
    print("Found %d objects." % len(result_out["detection_scores"]))

    image_with_boxes = draw_boxes(
        np.array(image_out), result_out["detection_boxes"],
        result_out["detection_class_entities"], result_out["detection_scores"],
        min_score=0.2
    )

    return Image.fromarray(image_with_boxes, 'RGB')




# We can use url_for('foo_view') for reverse-lookups in templates or view functions
@app.route('/raw/<z>/<x>/<y>.png')
def raw(z, x, y):
    qk = tileXY_to_quadkey(int(x), int(y), int(z))
    url = BASEURL.format(qk)
    print(url)


    with request.urlopen(url) as file:
        raw_img = Image.open(file)
        pred_img = do_inference(raw_img)

        return serve_pil_image(pred_img)




if __name__ == '__main__':
    app.run()
