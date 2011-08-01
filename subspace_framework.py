##    Author:   Diego Sona <sona@fbk.eu>
##    Created:  Dec 2010
##    Modified: Aug 2011
##    Version:  0.3
"""

Module Description
==================

Module providing the framework for Subspace Methods (SM).  The
framework is implemented providing both randomized and SearchLight
instances.  In the SearchLight approach a sphere spans all voxels over
the 3D brain data and the voxels in the sphere are used in a
train/test cycle.  In case of Random SM (RSM) the sphere is replaced
with a random selection of voxels.  The approach computes the
relevance voxels with various metods (precisions , sensitivities,
etc.).


Module Organization
===================

The module contains a class  providing both Seachlight
and RSM implementations

It contains a super class SubspaceMethods providing the basics for
the SM framework and as many classes inheriting from the super class
as the subspace methods implemented.
""" 

import sys
import time
import random
import numpy as np

from mvpa.datasets.splitters import NFoldSplitter



######################################################################
class SubspaceMethods(object): 
    """The class provide the support for searchlight and for random
    subspace generation of relervance maps.  Both algorithms span the
    entire brain.  While searchlight select the sphere of voxels
    surounding the analyzed one, the random subspace selects randomly
    the voxels in the whole brain.  Searchlight spans the brain only
    once, while random subspace spans the brain as many times as
    needed for a statistical robust result.
    """
   
    ##################################################
    def __init__(self, classifier, splitter=NFoldSplitter(), subspace='random', size=1.0): 
        """Initialization method

        :Parameters:
          classifier : an object of class "classifiers"
            Provides the classifier to be used in cross-validation.
            [currently only RidgeReg is supported]
          splitter : Splitter
            Used to split the dataset for cross-validation. By
            convention, for each pair in the touple, the first dataset
            is used to train the classifier.  The second dataset is
            used to generate predictions with the trained classifier.
            In case of NoneSplitter() training and test are performed
            with the same dataset returned as second element in the
            touple [default is NFoldSplitter()].
          subspace : 'random' | 'searchlight'
            The type of subspace selector [default is 'random'].
          size : real
            The number of random features to iteratively select in RSM
            (only the integer component will be used) or the radius in
            millimiters in searchlight [default is 1.0]
        """ 

        #if 'has_sensitivity' not in classifier._clf_internals:
        #    raise ValueError, "Classifier %s has not sensitivity analysis method" % classifier
        if 'ridge' not in classifier._clf_internals:
            raise ValueError, "Currently only Ridge regression classifier is supported in RSM"

        self.__classifier = classifier        # The classifier (currently only RidgeReg is suported
        self.__splitter = splitter            # The type of cross-validation [default NFoldSplitter()]
        self.__sensitivities = None           # Per-feature list of all sensitivities conputed in each run
        self.__precisions = None              # Per-feature list of precisions for each run
        self.__counter = None                 # Number of times a feature appeared in a sample
        self.__iters = 1                      # 
        self.__selection_policy = subspace            # The selection criterion
        if self.__selection_policy == 'random':
            self._selectIDs = self._random_selection
            self.__samplesize = int(size)        # Dimension of the set of sampled features
        elif self.__selection_policy == 'searchlight':
            self._selectIDs = self._searchlight_selection
            self.__radius = size
        else:
            raise ValueError("The features selector should either be 'random' or 'searchlight'")
        pass
    

        
    ##################################################
    def _random_selection(self, dataset, fID):
        """Randomly select a number of features from the dataset and return
        the features' identifiers.  The passed feature identifier is
        forced in the list.

        :Parameters:
          dataset : an object of class "dataset"
            It will not be used in the method.  It is there just
            because of compliance to the method call.
          fID : integer
            The feature identifier will be forced to be in the list of
            the selected features
            
        :Return:
          list : a sorted list of integers
            The selected features' identifiers
        """

        selectedIDs = random.sample(xrange(self.__featNumber), self.__samplesize)
        if fID not in selectedIDs:
            selectedIDs[0] = fID
            pass
        return sorted(selectedIDs)
        


    ##################################################
    def _searchlight_selection(self, dataset, fID):
        """Returns the identifier of the set of features in the neighborhood
        of the passed feature identifier.

        :Parameters:
          dataset : a dataset
            It will be used to get the neighborhood of central feature.
          fID : integer
            Identifier of the feature in the sphere center
            
        :Return:
          list : a sorted list of integers
            The selected features' identifiers
        """
        selectedIDs = dataset.mapper.getNeighbors(fID, self.__radius)
        return sorted(selectedIDs)


    ##################################################
    def set_iters(self, iters):
        """Used to set the number of iterations.  If not used the default is 1
           (see __init__)

        :Parameters:
          iters : integer
            The number of iterations.
        """
        self.__iters = iters
        pass
    
    

    ##################################################
    def _pre_call(self, dataset):
        """In the first call creates the data structure to preserve
        all the results.

        :Parameters:
          dataset : Datasets
        """
        if self.__counter == None:
            self.__featNumber = dataset.nfeatures
            self.__counter = np.zeros(self.__featNumber)
            self.__sensitivities = [[] for f in xrange(self.__featNumber)]       # Sensitivity of the current feature
            self.__precisions = [[] for f in xrange(self.__featNumber)]         # Precisions with the current feature
            pass
        pass



    ##################################################
    def __call__(self, dataset): 
        """Perform the analysis with random subsampling. 
        :Parameters: 
          dataset : Dataset
            Data to be used to train and test the classifier on random subspace samplings
        :Returns:
          (list, list)
            The first element of the tuple contains the the lists of
            the sensitivities computed for each feature for each run. The
            second element contains the list of precisions for
            each feature for each run.
        """
        # During the first call creates the data structure to preserve results        
        self._pre_call(dataset)

        startime = time.time()
        iteration = 0 # a counter of the number of iterations. For searchlight only one is needed.
        
        # Repeat as many time as need for RSM and only once for SL
        while iteration < self.__iters:
            for featID in xrange(self.__featNumber):
                if self.__selection_policy == 'random' and self.__counter[featID] > iteration:
                    continue   # skip the current feature because already used more than needed (just with RSM)
                selectedIDs = self._selectIDs(dataset, featID) # select the features either randomly or in the neighborhood
                datasubspace = dataset.selectFeatures(selectedIDs, sort=False, plain=True) # project the data in the new feature space
                
                # execute the train/test cicle with cross-validation
                tmp_corrects = 0
                weights = np.zeros(len(selectedIDs)) # Sensitivities
                
                for spl in self.__splitter(datasubspace):
                    testset = spl[1]
                    if spl[0] != None: trainset = spl[0]  # Cross-validation is adopted
                    else:              trainset = testset # Biased evaluation is adopted
                    
                    self.__classifier.train(trainset)                    
                    prediction = np.sign(self.__classifier.predict(testset.samples))
                    weights += self.__classifier.w[:-1] # remove the last bias weight. This is good with RidgeReg only.  getSensitivities() should be used indeed
                    
                    tmp_corrects += np.sum(prediction == np.asarray(testset.labels))
                    pass
                tmp_precision = tmp_corrects / float(datasubspace.nsamples)
                
                # Store the sensitivities and the number of correct clasifications for all selected labels in RSM.
                # In SL save the precision only for voxels central to the sphere.
                if self.__selection_policy == 'random':
                    for f, fID in enumerate(selectedIDs):
                        self.__sensitivities[fID].append(weights[f])
                        self.__precisions[fID].append(tmp_precision)
                        self.__counter[fID] += 1
                        pass
                elif self.__selection_policy == 'searchlight':
                    for f, fID in enumerate(selectedIDs):
                        self.__sensitivities[fID].append(weights[f])
                        pass
                    self.__precisions[featID].append(tmp_precision)
                    self.__counter[featID] += 1
                else:
                    raise ValueError("The subspace selection must be either 'random' or 'searchlight'")
            
                if __debug__:
                    s = time.time()-startime
                    h = int(np.floor(s / 3600))
                    m = int(np.floor((s - h*3600) / 60))
                    s = int(s - m*60 - h*3600)
                    outstr = "\rIteration-feature: %5d-%5d \tSelected features counter (average, max):%.2f-%3d - Elapsed time: %3dh %2d' %2d''" \
                        % (iteration, featID, self.__counter.mean(), self.__counter.max(), h, m, s)
                    sys.stdout.write(outstr)
                    sys.stdout.flush()
                    pass
                pass
            iteration += 1
            pass
        
        return (self.__sensitivities, self.__precisions)
    

