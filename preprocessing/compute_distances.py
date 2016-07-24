# -*- coding: utf-8 -*-
###################################################################################################
# Copyright (c) 2016 , Regents of the University of Colorado on behalf of the University          #
# of Colorado Colorado Springs.  All rights reserved.                                             #
#                                                                                                 #
# Redistribution and use in source and binary forms, with or without modification,                #
# are permitted provided that the following conditions are met:                                   #
#                                                                                                 #
# 1. Redistributions of source code must retain the above copyright notice, this                  #
# list of conditions and the following disclaimer.                                                #
#                                                                                                 #
# 2. Redistributions in binary form must reproduce the above copyright notice, this list          #
# of conditions and the following disclaimer in the documentation and/or other materials          #
# provided with the distribution.                                                                 #
#                                                                                                 #
# 3. Neither the name of the copyright holder nor the names of its contributors may be            #
# used to endorse or promote products derived from this software without specific prior           #
# written permission.                                                                             #
#                                                                                                 #
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY             #
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF         #
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL          #
# THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,            #
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF     #
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)          #
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,           #
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS           #
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                                    #
#                                                                                                 #
#                                                                                                 #
# If you use this code, please cite the following works                                           #
#                                                                                                 #
# A. Bendale, T. Boult “Towards Open Set Deep Networks” IEEE Conference on                        #
# Computer Vision and Pattern Recognition (CVPR), 2016                                            #
#                                                                                                 #
# Notice Related to using LibMR.                                                                  #
#                                                                                                 #
# If you use Meta-Recognition Library (LibMR), please note that there is a                        #
# difference license structure for it. The citation for using Meta-Recongition                    #
# library (LibMR) is as follows:                                                                  #
#                                                                                                 #
# Meta-Recognition: The Theory and Practice of Recognition Score Analysis                         #
# Walter J. Scheirer, Anderson Rocha, Ross J. Micheals, and Terrance E. Boult                     #
# IEEE T.PAMI, V. 33, Issue 8, August 2011, pages 1689 - 1695                                     #
#                                                                                                 #
# Meta recognition library is provided with this code for ease of use. However, the actual        #
# link to download latest version of LibMR code is: http://www.metarecognition.com/libmr-license/ #
###################################################################################################


import scipy as sp
import sys
import os, glob
import os.path as path
import scipy.spatial.distance as spd
from scipy.io import loadmat, savemat

featurefilepath = '../data/train_features/'

#------------------------------------------------------------------------------------------
def getlabellist(fname):
    """ Read synset file for ILSVRC 2012
    """

    imagenetlabels = open(fname, 'r').readlines()
    labellist  = [i.split(' ')[0] for i in imagenetlabels]        
    return labellist

#------------------------------------------------------------------------------------------
def compute_channel_distances(mean_train_channel_vector, features, category_name):
    """
    Input:
    ---------
    mean_train_channel_vector : mean activation vector for a given class. 
                                It can be computed using MAV_Compute.py file
    features: features for the category under consideration
    category_name: synset_id

    Output:
    ---------
    channel_distances: dict of distance distribution from MAV for each channel. 
    distances considered are eucos, cosine and euclidean
    """

    eucos_dist, eu_dist, cos_dist = [], [], []
    for channel in range(features[0].shape[0]):
        eu_channel, cos_channel, eu_cos_channel = [], [], []
        # compute channel specific distances
        for feat in features:
            eu_channel += [spd.euclidean(mean_train_channel_vector[channel, :], feat[channel, :])]
            cos_channel += [spd.cosine(mean_train_channel_vector[channel, :], feat[channel, :])]
            eu_cos_channel += [spd.euclidean(mean_train_channel_vector[channel, :], feat[channel, :])/200. +
                               spd.cosine(mean_train_channel_vector[channel, :], feat[channel, :])]
        eu_dist += [eu_channel]
        cos_dist += [cos_channel]
        eucos_dist += [eu_cos_channel]

    # convert all arrays as scipy arrays
    eucos_dist = sp.asarray(eucos_dist)
    eu_dist = sp.asarray(eu_dist)
    cos_dist = sp.asarray(cos_dist)

    # assertions for length check
    assert eucos_dist.shape[0] == 10
    assert eu_dist.shape[0] == 10
    assert cos_dist.shape[0] == 10
    assert eucos_dist.shape[1] == len(features)
    assert eu_dist.shape[1] == len(features)
    assert cos_dist.shape[1] == len(features)

    channel_distances = {'eucos': eucos_dist, 'cosine': cos_dist, 'euclidean':eu_dist}
    return channel_distances
    
#------------------------------------------------------------------------------------------
def compute_distances(mav_fname, labellist, category_name, 
                      featurefilepath, layer = 'fc8'):
    """
    Input:
    -------
    mav_fname : path to filename that contains mean activation vector
    labellist : list of labels from ilsvrc 2012
    category_name : synset_id

    """
    
    
    mean_feature_vec = loadmat(mav_fname)[category_name]
    print '%s/%s/*.mat' %(featurefilepath, category_name)
    featurefile_list = glob.glob('%s/*.mat' %featurefilepath)

    correct_features = []
    for featurefile in featurefile_list:
        try:
            img_arr = loadmat(featurefile)
            predicted_category = labellist[img_arr['scores'].argmax()]
            if predicted_category == category_name:
                correct_features += [img_arr[layer]]
        except TypeError:
            continue

    distance_distribution = compute_channel_distances(mean_feature_vec, correct_features, category_name)
    return distance_distribution

#------------------------------------------------------------------------------------------
def main():

    if len(sys.argv[1:]) != 3:
        print "usage: python compute_distances.py <synset id> <mav_file> <feature_path> \n(e.g.)"
        print "python compute_distances.py n01440764 ../data/mean_files/n01440764.mat ../data/train_features/n01440764/"
    else:
        category_name = sys.argv[1]
        mav_fname = sys.argv[2]
        feature_path = sys.argv[3]
        labellist = getlabellist('../synset_words_caffe_ILSVRC12.txt')

        distance_distribution = compute_distances(mav_fname, labellist, category_name, feature_path)
        savemat('%s_distances.mat'%category_name, distance_distribution)

if __name__ == "__main__":
    main()
