##    Author:   Diego Sona <sona@fbk.eu>
##    Created:  Dec 2010
##    Modified: Jul 2011
##    Version:  0.3

##    Description: 

##    This piece of code shows how to use the RSM framework on a fMRI
##    dataset.  In particular, in this implemenetation the Haxby
##    dataset has been used with a specific task (faces vs houses).
##    The dataset on subject 1 only is loaded and transformed in a
##    binary classification problem.

import os
import numpy as np

from mvpa.suite import RidgeReg, NiftiDataset
from mvpa.datasets.splitters import NFoldSplitter

from subspace_framework import SubspaceMethods

labels_dict = {0:'rest',
               1:'cat',
               2:'face',
               3:'shoe',
               4:'house',
               5:'chair',
               6:'bottle',
               7:'scissors',
               8:'scrambledpix'
               }

labels_index = {'rest'        :0,
                'cat'         :1,
                'face'        :2,
                'shoe'        :3,
                'house'       :4,
                'chair'       :5,
                'bottle'      :6,
                'scissors'    :7,
                'scrambledpix':8
                }

experiment1 = {'description': 'Face-House',
               'posClass':    ['face'],
               'negClass':    ['house']
               }


#################################################
def remodulate_data(data, experiment, shift=2, stretch=1):
    '''Changes the labeling and chunking according to the experiment and
    the expected hemodynamics
    :Parameters: 
      data : object of class Dataset
        Data that will be transformed in a two-class problem shifting
        and streching the experiment according to the expected
        hemodynamics
      experiment : dictionary
        The structure describes the experiment defining positive and
        negative classes with the keys "posClass" and "negClass"
      shift : integer
        Indicates how many TR the labels should be shifted in order to
        allign the measure BOLD activity with the expected sequence of
        stimuly.  Usually 4 secs are good, i.e, generally 2 TRs.
        [default is 2]
      stretch : integer
        Indicates how many volumes the labels should be stretched
        after the peak.  This increases the number of volmes can be
        used for any experiment including those that are in the
        descendant part of HRF curve.  [default is 1]

    :Returns:
      newdata : object of class Dataset
        A dataset with binary classification task and with labels
        shifted and stretched.
    '''

    newdata = data.copy()

    # Modify the labels into a two-class problem determining negative
    # and positive samples, and samples to be removed.
    for k,v in labels_dict.iteritems():
        if v in experiment['negClass']:
            newdata.labels[data.labels == k] = -1
        elif v in experiment['posClass']:
            newdata.labels[data.labels == k] = +1
        else:
            newdata.labels[data.labels == k] = 0
            pass
        pass
    
    # Shifting the labels and the chunks according to the hemodynamic
    # lag (time to peak) 
    shift = 2 # conservative
    newdata.labels = np.roll(newdata.labels, shift)
    newdata.chunks = np.roll(newdata.chunks, shift)
    for i in range(shift):
        newdata.labels[i] = newdata.labels[shift]
        newdata.chunks[i] = newdata.chunks[shift]
        pass
    
    # Stretching the stimuli in time replicating many times the last
    # stimulus before resting due to hemodynaimc lag (usually 2-4 secs
    # are good, i.e., 1-2 TRs)
    stretch = 1  # conservative
    for j in range(stretch):
        for i in range(newdata.labels.size-1, 1+j, -1) :
            if newdata.labels[i]==0 and newdata.labels[i-1]!=0:
                newdata.labels[i]=newdata.labels[i-1]
                pass
            pass
        pass
    
    # Useless data are removed from the dataset
    newdata = newdata.selectSamples(newdata.labels != 0)
    
    return newdata



##################################################
if __name__=='__main__':

    baseroot = "/home/sona/Development/Datasets/Haxby_datasets/subj1"   # <directory where data is located ...... new data will be saved here as well>
    datafilename  = os.path.join(baseroot, 'bold.nii.gz')         # fMRI data
    maskfilename  = os.path.join(baseroot, 'brain_mask.nii.gz')   # mask to avoid computations outside the brain - any other mask for ROIs can be used instead
    labelfilename = os.path.join(baseroot, 'labels.txt')          # labels
    

    classifier = RidgeReg()       # Currently the RSM framework only support Ridge Regression for theoretical reasons.
    sample_size = 150              # Either the number of features to be sampled with RSM or the radius in millimeters for SearchLight
    max_rsm_samples = 200         # Number of iterations over the brain, i.e., minimum number of times a voxel is selected
    samplingMethod = 'random'     # It could be 'random' or 'searchlight'

    ##################################################
    # Loading data
    # Loading the labels of Haxby dataset http://dev.pymvpa.org/datadb/haxby2001.html
    f = open(labelfilename)
    raw_labelsChunks = f.read().split('\n') # read the file spliting the rows
    f.close()
    attribute_labels = []
    attribute_chunks = []
    # Reading the (label, chunk) pairs skipping the header.
    for i in range(len(raw_labelsChunks)):
        if 'labels' not in raw_labelsChunks[i] and len(raw_labelsChunks[i])>0: # skip the header and empty lines
            attribute_labels.append(labels_index[raw_labelsChunks[i].split()[0]])
            attribute_chunks.append(map(int, [raw_labelsChunks[i].split()[1]]))
            pass
        pass

    dataset = NiftiDataset(samples=datafilename, mask=maskfilename, labels=attribute_labels, chunks=attribute_chunks)
    print 'Loaded dataset:\n\t', dataset


    ##################################################
    # Preprocessing
    print "Detrending"
    dataset.detrend(perchunk=False, model='regress', polyord=3)  # Polynimial detrending order 3 on the entire timeseries
    print "Normalizing pervoxel"
    dataset.zscore(perchunk=False, pervoxel=True)  # z-score normalization on the entire timeseries per voxels singularly


    ##################################################
    # Encoding of experiment
    print 'Remodulating the data according to the selected experiment'
    dataset = remodulate_data(dataset, experiment1) 
    print 'Encoded dataset:\n\t', dataset
    
    
    ##################################################
    # Execution of the selected method
    mapper = SubspaceMethods(classifier, splitter=NFoldSplitter(), subspace=samplingMethod, size=sample_size)
    mapper.set_iters(max_rsm_samples)
    (sensitivities, precisions) = mapper(dataset)

    # Averaging the computed measures over all runs (usefull in case of RSM)
    sens_map = []
    prec_map = []
    for i in range(len(sensitivities)):
        sens_map.append(np.average(sensitivities[i]))
        prec_map.append(np.average(precisions[i]))
        pass
    
    ##################################################
    # Saving the results in nifty format
    dataset.map2Nifti(sens_map).save(os.path.join(baseroot, samplingMethod+'_'+str(sample_size)+'sensitivity_map-average_'+str(max_rsm_samples)+'_samples.nii.gz'))
    dataset.map2Nifti(prec_map).save(os.path.join(baseroot, samplingMethod+'_'+str(sample_size)+'precision_map-average_'+str(max_rsm_samples)+'_samples.nii.gz'))
    
