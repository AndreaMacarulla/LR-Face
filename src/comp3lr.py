"""Performs face alignment and calculates L2 distance between the embeddings of images al LR."""

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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import tensorflow as tf 
import numpy as np
import sys
import os
import copy
import argparse
import facenet 
import align.detect_face
import xlsxwriter 
#import imageio

def main(args):
    
    a_same = args.a_same
    a_different = args.a_different
    b_same = args.b_same
    b_different = args.b_different
    outfile = args.out_file
    

    images = load_and_align_data(args.image_files, args.image_size, args.margin, args.gpu_memory_fraction)
    with tf.Graph().as_default():

        with tf.Session() as sess:
      
            # Load the model
            facenet.load_model(args.model)
    
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            # Run forward pass to calculate embeddings
            feed_dict = { images_placeholder: images, phase_train_placeholder:False }
            emb = sess.run(embeddings, feed_dict=feed_dict)
            
            nrof_images = len(args.image_files)
            
            # Create output folder if it doesn't exist
            outf = os.path.dirname(outfile)
            if not os.path.exists(outf):
                os.makedirs(outf)
                
            #fd = open(outfile,'w')
            workbook = xlsxwriter.Workbook(outfile)
            worksheet = workbook.add_worksheet("Comparisons")
            
            row = 0
            bold = workbook.add_format({'bold': True})
            worksheet.write(row, 0, 'Image 1', bold)
            worksheet.write(row, 1, 'Image 2', bold)
            worksheet.write(row, 2, ' Distance',bold)
            worksheet.write(row, 3,  'LR', bold)
            row+=1
            
            

            print ('RESULTS:')
            print 
            for i in range(nrof_images-1):
                
                for j in range (i+1,nrof_images):
                    dist = np.sqrt(np.sum(np.square(np.subtract(emb[i,:], emb[j,:]))))
                    same = weibull(dist,a_same,b_same)
                    different = weibull(dist,a_different,b_different)
                    lr = same/different
                    #fd.write('%s: %s: %1.4f: %3.4f \n' % (args.image_files[i], args.image_files[j], dist, lr))
                    worksheet.write(row, 0, os.path.basename(args.image_files[i]))
                    worksheet.write(row, 1, os.path.basename(args.image_files[j]))
                    worksheet.write(row, 2,  dist)
                    worksheet.write(row, 3,  lr)
                    row+=1
                    
            print ('END')
            
            #fd.close
            workbook.close()
            
            
def load_and_align_data(image_paths, image_size, margin, gpu_memory_fraction):

    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor
    
    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
  
    tmp_image_paths=copy.copy(image_paths)
    img_list = []
    for image in tmp_image_paths:
        img = misc.imread(os.path.expanduser(image), mode='RGB')
#        img = imageio.imread(os.path.expanduser(image))

        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        if len(bounding_boxes) < 1:
          image_paths.remove(image)
          print("can't detect face, remove ", image)
          continue
        det = np.squeeze(bounding_boxes[0,0:4])
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0]-margin/2, 0)
        bb[1] = np.maximum(det[1]-margin/2, 0)
        bb[2] = np.minimum(det[2]+margin/2, img_size[1])
        bb[3] = np.minimum(det[3]+margin/2, img_size[0])
        cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
        aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
#       aligned = np.array(cropped.resize (image_size, image_size), interp='bilinear')
        prewhitened = facenet.prewhiten(aligned)
        
        img_list.append(prewhitened)
    images = np.stack(img_list)
    return images

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('model', type=str, 
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('image_files', type=str, nargs='+', help='Images to compare')
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--margin', type=int,
        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=0.8)
    parser.add_argument('--a_same', type=float,
        help='Scale parameter (a) for Weibull, same person.', default=0.796174014389305)
    parser.add_argument('--b_same', type=float,
        help='Shape parameter (b) for Weibull, same person.', default=3.750398608516835)
    parser.add_argument('--a_different', type=float,
        help='Scale parameter (a) for Weibull, different person.', default=1.405668708882522)
    parser.add_argument('--b_different', type=float,
        help='Shape parameter (b) for Weibull, different person.', default=12.478248305735089)
    parser.add_argument('--out_file', type=str,
        help='Output file.', default='../output/results.txt')

#Weibull different
#a = 1.405668708882522
#b = 12.478248305735089

#Weibull same
#a = 0.796174014389305
#b = 3.750398608516835

                
            
    return parser.parse_args(argv)
    
def weibull(x,a,b):
    x2 = x/a
    p1 = x2**b
    y = b/a * x2**(b-1) * np.exp(-p1)
    return(y)



    
    
def launcher(path_images,output_file):
    
    argv = ['../data/model/20180402-114759.pb']
    
    for root,dirL,fileL in os.walk(path_images):
        for file in fileL:
            argv.append('/'.join([root,file]))
            
    argv.append('--out_file')
    argv.append(output_file)
    argv.append('--gpu_memory_fraction')
    argv.append('0.75')        
               
    
    
    main(parse_arguments(argv))    
    

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
