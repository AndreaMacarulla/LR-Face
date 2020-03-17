"""Performs face alignment and calculates L2 distance between the embeddings of images al LR.
https://github.com/davidsandberg/facenet
"""

# MIT License
#
# Copyright (c) 2016 David Sandberg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from contextlib import contextmanager
import copy
from pathlib import Path
import os

import facenet
import numpy as np
from scipy import misc
import tensorflow as tf

import detect_face


def calculate_face_likelihood(input_path, output_path, reference_image, image_size, margin,
                              gpu_memory_fraction, a_same, a_different, b_same, b_different):
    """

    :param input_path: path to directory of images of faces
    :param output_path: path to text file where results will be stored
    :param reference_image: path to image to compare againse
    :param model_path: path to pretrained tensorflow model
    :param image_size: size of images
    :param margin: margin of images
    :param gpu_memory_fraction: how much gpu memory to use
    :param a_same:
    :param a_different:
    :param b_same:
    :param b_different:
    :return:
    """
    sess = tf.get_default_session()

    image_files = [reference_image, input_path]
    images = load_and_align_data(image_files, image_size, margin, gpu_memory_fraction)

    # Get input and output tensors
    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

    # Run forward pass to calculate embeddings
    feed_dict = {images_placeholder: images, phase_train_placeholder: False}
    emb = sess.run(embeddings, feed_dict=feed_dict)

    # Create output folder if it doesn't exist
    outf = os.path.dirname(output_path)
    if not os.path.exists(outf):
        os.makedirs(outf)

    dist = np.sqrt(np.sum(np.square(np.subtract(emb[0, :], emb[1, :]))))
    same = weibull(dist, a_same, b_same)
    different = weibull(dist, a_different, b_different)
    lr = same / different
    name1, name2 = Path(image_files[0]).name, Path(image_files[1]).name

    with open(output_path, 'w') as fd:
        fd.write('%s, %s, %1.4f, %3.4f \n' % (name1, name2, dist, lr))

    return {
        'image': input_path,
        'query': reference_image,
        'distance': float(dist),
        'likelihood_ratio': float(lr),
    }


def load_and_align_data(image_paths, image_size, margin, gpu_memory_fraction):
    """ Detect Faces, ignore images where faces do not occur. Then crop, align and whiten images.
    :param image_paths:
    :param image_size:
    :param margin:
    :param gpu_memory_fraction:
    :return: List of images which have been processed
    """
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

    tmp_image_paths = copy.copy(image_paths)
    img_list = []
    for image in tmp_image_paths:
        img = misc.imread(os.path.expanduser(image), mode='RGB')

        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        if len(bounding_boxes) < 1:
            image_paths.remove(image)
            print("can't detect face, remove ", image)
            continue
        det = np.squeeze(bounding_boxes[0, 0:4])
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0]-margin/2, 0)
        bb[1] = np.maximum(det[1]-margin/2, 0)
        bb[2] = np.minimum(det[2]+margin/2, img_size[1])
        bb[3] = np.minimum(det[3]+margin/2, img_size[0])
        cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
        aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
        pre_whitened = facenet.prewhiten(aligned)

        img_list.append(pre_whitened)
    images = np.stack(img_list)
    return images


def weibull(x, a, b):
    """Calculate weibull distribution.
    # https://en.wikipedia.org/wiki/Weibull_distribution
    :param x: parameter
    :param a: hyper parameter
    :param b: hyper parameter
    :return:
    """

    x2 = x/a
    p1 = x2**b
    y = b/a * x2**(b-1) * np.exp(-p1)
    return y


@contextmanager
def context():
    with tf.Graph().as_default():

        with tf.Session() as sess:
            # Load the model
            facenet.load_model('model.pb')
            yield


def face_verification_likelihood(input_path, output_path, reference_image, **kwargs):
    """Takes a directory of images and compares each image to every other image.
    For each image comparison produces liklihood ration that the images are of the same face.
    Assumes images are of faces.
    Stored result in output_file (text file).

    :param input_path: path to folder of face images
    :param output_path: path to text file of results
    :param reference_image: path to image to compare againse
    :param model_path: path to tensorflow model
    """

    args = {
        'image_size': 160,
        'margin': 44,
        'gpu_memory_fraction': 0.75,
        'a_same': 0.796174014389305,
        'b_same': 3.750398608516835,
        'a_different': 1.405668708882522,
        'b_different': 12.478248305735089,
    }
    result = calculate_face_likelihood(input_path=input_path, output_path=output_path, reference_image=reference_image,
                                       **args)
    return result
