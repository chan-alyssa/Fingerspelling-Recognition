# Fingerspelling-Recognition

This project has 2 objectives:
1. To identify if fingerspelling is occuring in continuously signed videos
2. If fingerspelling is present, identify the letters being spelt

To process the data, a CSV containing a list of fingerspelt words, start and end times is input into the hamerprocessing.py script. This produces an H5 file which can then be input into extractfeatures.py to extract the features. The features consist of 2D and 3D keypoints and forms the training and testing data. The data contains instances of fingerspelling, lexical signs, gestures and hands at rest. 

Once you have the testing and training data, binaryclassifier.py or binarybatch.py can be used to train the classifier. This will result in approximately 88% per frame accuracy.

Below is a brief description of the python files and the inputs/outputs.

Binarybatch.py:
Trains a neural network for fingerspelling identification. Batches the training data. Input both train and test file. 

Binaryclassifier.py:
Trains a neural network for fingdrspelling identification. Does not batch the training data. Oversamples the minority class. 

Binaryvisualisation.py:
Input h5 file containing the processed hand information from the HaMer model. Outputs the image on the top and two graphs indicating if fingerspelling is present - one contains the raw predictions and the other a smoothed version. 

Cleanedautomaticannotations.csv:
CSV containing intervals where fingerspelling is detected. These annotations were originally from the Transpeller project. We have used the classifier to clean up the intervals so that the fingerspelling events are more exact. Other issues such as no fingerspelling detected or multiple events are also recorded in this file. 

Extractfeatures.py:
Input h5 file from hammerprocessing.py that contains hand position information. Extracts 86 features (2D keypoints) for binary classification and 384 features (3D keypoints) for letter classification. 

Hamerprocessing.py:
Input a CSV file with the video ID and fingerspelling events recorded and outputs an h5 file with hand information. 
