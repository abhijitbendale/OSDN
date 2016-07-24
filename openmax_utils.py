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
# Author: Abhijit Bendale (abendale@vast.uccs.edu)                                                #
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


import os, sys, pickle, glob
import os.path as path
import argparse
import scipy.spatial.distance as spd
import scipy as sp


def parse_synsetfile(synsetfname):
    """ Read ImageNet 2012 file
    """
    categorylist = open(synsetfname, 'r').readlines()
    imageNetIDs = {}
    count = 0
    for categoryinfo in categorylist:
        wnetid = categoryinfo.split(' ')[0]
        categoryname = ' '.join(categoryinfo.split(' ')[1:])
        imageNetIDs[str(count)] = [wnetid, categoryname]
        count += 1

    assert len(imageNetIDs.keys()) == 1000
    return imageNetIDs

def getlabellist(synsetfname):
    """ read sysnset file as python list. Index corresponds to the output that 
    caffe provides
    """
    
    categorylist = open(synsetfname, 'r').readlines()
    labellist = [category.split(' ')[0] for category in categorylist]
    return labellist


def compute_distance(query_channel, channel, mean_vec, distance_type = 'eucos'):
    """ Compute the specified distance type between chanels of mean vector and query image.
    In caffe library, FC8 layer consists of 10 channels. Here, we compute distance
    of distance of each channel (from query image) with respective channel of
    Mean Activation Vector. In the paper, we considered a hybrid distance eucos which
    combines euclidean and cosine distance for bouding open space. Alternatively,
    other distances such as euclidean or cosine can also be used. 
    
    Input:
    --------
    query_channel: Particular FC8 channel of query image
    channel: channel number under consideration
    mean_vec: mean activation vector

    Output:
    --------
    query_distance : Distance between respective channels

    """

    if distance_type == 'eucos':
        query_distance = spd.euclidean(mean_vec[channel, :], query_channel)/200. + spd.cosine(mean_vec[channel, :], query_channel)
    elif distance_type == 'euclidean':
        query_distance = spd.euclidean(mean_vec[channel, :], query_channel)/200.
    elif distance_type == 'cosine':
        query_distance = spd.cosine(mean_vec[channel, :], query_channel)
    else:
        print "distance type not known: enter either of eucos, euclidean or cosine"
    return query_distance
    
