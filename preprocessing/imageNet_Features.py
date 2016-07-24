##########################################################################################################################################################################
# This file is adapted from Caffe's classify demo found at                                                                                                               #
# https://github.com/BVLC/caffe/blob/master/python/classify.py                                                                                                           #
# The original file was Caffe: a fast open framework for deep learning. http://caffe.berkeleyvision.org/                                                                 #
# For original license please check https://github.com/BVLC/caffe                                                                                                        #
# If you use this file, please consider citing                                                                                                                           #
# @article{jia2014caffe,                                                                                                                                                 #
#   Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor}, #
#   Journal = {arXiv preprint arXiv:1408.5093},                                                                                                                          #
#   Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},                                                                                              #
#   Year = {2014}                                                                                                                                                        #
# }                                                                                                                                                                      #
##########################################################################################################################################################################


import scipy as sp
import os, sys, glob
import os.path as path
import caffe
import argparse, time
import pandas as pd
from skimage.color import rgb2gray
import numpy as np
from scipy.io import savemat

import multiprocessing as mp

NPROCESSORS  = 31

def runClassifierTest(args):
    """ Given list of arguments, run classifier
    """

    image_dims = [int(s) for s in args.images_dim.split(',')]
    if args.force_grayscale:
      channel_swap = None
      mean_file = None
    else:
      channel_swap = [int(s) for s in args.channel_swap.split(',')]
      mean_file = args.mean_file

    # Make classifier.
    classifier = caffe.Classifier(args.model_def, args.pretrained_model,
            image_dims=image_dims, gpu=args.gpu, mean_file=mean_file,
            input_scale=args.input_scale, channel_swap=channel_swap)

    if args.gpu:
        print 'GPU mode'

    # Load numpy array (.npy), directory glob (*.jpg), or image file.
    args.input_file = os.path.expanduser(args.input_file)
    if args.input_file.endswith('npy'):
        inputs = np.load(args.input_file)
    elif os.path.isdir(args.input_file):
        inputs =[caffe.io.load_image(im_f)
                 for im_f in glob.glob(args.input_file + '/*.' + args.ext)]
    else:
        inputs = [caffe.io.load_image(args.input_file)]

    if args.force_grayscale:
      inputs = [rgb2gray(input) for input in inputs];

    print "Classifying %d inputs." % len(inputs)

    # Classify.
    start = time.time()
    scores = classifier.predict(inputs, not args.center_only).flatten()
    print "Done in %.2f s." % (time.time() - start)

    if args.print_results:
        with open(args.labels_file) as f:
          labels_df = pd.DataFrame([
               {
                   'synset_id': l.strip().split(' ')[0],
                   'name': ' '.join(l.strip().split(' ')[1:]).split(',')[0]
               }
               for l in f.readlines()
            ])
        labels = labels_df.sort('synset_id')['name'].values

        indices = (-scores).argsort()[:5]
        predictions = labels[indices]

        meta = [
                   (p, '%.5f' % scores[i])
                   for i, p in zip(indices, predictions)
               ]

        print meta

    # Save
    np.save(args.output_file, scores)

def obtainImgList(imgFileDir):
    """ Find all the images for which caffe features have to be extracted
    """


    train_images = glob.glob(path.join(imgFileDir, 'train/*/*.JPEG'))

    return train_images
    
def extractFeatures(args):
    """ Loop through files, extract caffe features, and save them to file
    """


    # Load numpy array (.npy), directory glob (*.jpg), or image file.
        
    imglist = obtainImgList('/home/ubuntu/datadir/imageNetForWeb/')
    imagelist_args = [int(s) for s in args.imagelist_args.split(',')]
    pool = mp.Pool(processes=NPROCESSORS)
    arg_l = []
    st = time.time()
    imglistFeatures = imglist[imagelist_args[0]:imagelist_args[1]]
    for imgname in imglistFeatures:
        arg_l += [(imgname, args)]
    pool.map(compute_features_multiproc, arg_l)
    print "Time taken for extracting features from %s images %s secs with %s Processors" %(len(imglistFeatures), time.time() - st, NPROCESSORS)


def compute_features(imgname, args):
    """
    Instantiate a classifier class, pass the images through the network and save features.
    Features are saved in .mat format
    """
    image_dims = [int(s) for s in args.images_dim.split(',')]
    if args.force_grayscale:
      channel_swap = None
      mean_file = None
    else:
      channel_swap = [int(s) for s in args.channel_swap.split(',')]
      mean_file = args.mean_file

    # Make classifier.
    classifier = caffe.Classifier(args.model_def, args.pretrained_model,
            image_dims=image_dims, gpu=args.gpu, mean_file=mean_file,
            input_scale=args.input_scale, channel_swap=channel_swap)

    if args.gpu:
        print 'GPU mode'



    outfname = imgname.replace('imageNetForWeb', 'imageNetForWeb_Features') + ".mat"
    print outfname
    if not path.exists(path.dirname(outfname)):
        os.makedirs(path.dirname(outfname))

    inputs = [caffe.io.load_image(imgname)]
    
    if args.force_grayscale:
        inputs = [rgb2gray(input) for input in inputs];

    print "Classifying %d inputs." % len(inputs)

    scores = classifier.predict(inputs, not args.center_only)
        # Now save features
    feature_dict = {}
    feature_dict['IMG_NAME'] = path.join(path.dirname(imgname), path.basename(imgname))
    feature_dict['fc7'] = sp.asarray(classifier.blobs['fc7'].data.squeeze(axis=(2,3)))
    feature_dict['fc8'] = sp.asarray(classifier.blobs['fc8'].data.squeeze(axis=(2,3)))
    feature_dict['prob'] = sp.asarray(classifier.blobs['prob'].data.squeeze(axis=(2,3)))
    feature_dict['scores'] = sp.asarray(scores)
    savemat(outfname, feature_dict)

def compute_features_multiproc(params):
    """ Multi-Processing interface for extarcting features
    """
    return compute_features(*params)

def main(argv):

    pycaffe_dir = os.path.dirname(__file__)

    parser = argparse.ArgumentParser()
    # Required arguments: input and output files.
    parser.add_argument(
        "input_file",
        help="Input image, directory, or npy."
    )
    parser.add_argument(
        "output_file",
        help="Output npy filename."
    )
    # Optional arguments.
    parser.add_argument(
        "--model_def",
        default=os.path.join(pycaffe_dir,
                "../examples/imagenet/imagenet_deploy.prototxt"),
        help="Model definition file."
    )
    parser.add_argument(
        "--pretrained_model",
        default=os.path.join(pycaffe_dir,
                "../examples/imagenet/caffe_reference_imagenet_model"),
        help="Trained model weights file."
    )
    parser.add_argument(
        "--gpu",
        action='store_true',
        help="Switch for gpu computation."
    )
    parser.add_argument(
        "--center_only",
        action='store_true',
        help="Switch for prediction from center crop alone instead of " +
             "averaging predictions across crops (default)."
    )
    parser.add_argument(
        "--images_dim",
        default='256,256',
        help="Canonical 'height,width' dimensions of input images."
    )
    parser.add_argument(
        "--mean_file",
        default=os.path.join(pycaffe_dir,
                             'caffe/imagenet/ilsvrc_2012_mean.npy'),
        help="Data set image mean of H x W x K dimensions (numpy array). " +
             "Set to '' for no mean subtraction."
    )
    parser.add_argument(
        "--input_scale",
        type=float,
        default=255,
        help="Multiply input features by this scale before input to net"
    )
    parser.add_argument(
        "--channel_swap",
        default='2,1,0',
        help="Order to permute input channels. The default converts " +
             "RGB -> BGR since BGR is the Caffe default by way of OpenCV."

    )
    parser.add_argument(
        "--ext",
        default='jpg',
        help="Image file extension to take as input when a directory " +
             "is given as the input file."
    )
    parser.add_argument(
        "--labels_file",
        default=os.path.join(pycaffe_dir,
                "../data/ilsvrc12/synset_words.txt"),
        help="Readable label definition file."
    )
    parser.add_argument(
        "--print_results",
        action='store_true',
        help="Write output text to stdout rather than serializing to a file."
    )
    parser.add_argument(
        "--force_grayscale",
        action='store_true',
        help="Converts RGB images down to single-channel grayscale versions," +
             "useful for single-channel networks like MNIST."
    )

    parser.add_argument(
        "--run_quick_test",
        action='store_true',
        help="Switch for gpu computation."
    )

    parser.add_argument(
        "--extract_features",
        action='store_true',
        help="Switch for gpu computation."
    )

    parser.add_argument(
        "--imagelist_args",
        default='0,20',
        help="images in imglist to consider for feature computation"
    )

    
    args = parser.parse_args()
    
    if args.run_quick_test:
        runClassifierTest(args)

    if args.extract_features:
        extractFeatures(args)

if __name__ == "__main__":
    main(sys.argv)
