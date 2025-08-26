# Fingerspelling-Recognition

Binarybatch.py
Trains a neural network for fingerspelling identification. Batches the training data. Input both train and test file. 

Binaryclassifier.py
Trains a neural network for fingdrspelling identification. Does not batch the training data. Oversamples the minority class. 

Binaryvisualisation.py
Input h5 file containing the processed hand information from the HaMer model. Outputs the image on the top and two graphs indicating if fingerspelling is present - one contains the raw predictions and the other a smoothed version. 

Cleanedautomaticannotations.csv
CSV containing intervals where fingerspelling is detected. These annotations were originally from the Transpeller project. We have used the classifier to clean up the intervals so that the fingerspelling events are more exact. Other issues such as no fingerspelling detected or multiple events are also recorded in this file. 

Extractfeatures.py
Input h5 file from hammerprocessing.py that contains hand position information. Extracts 86 features (2D keypoints) for binary classification and 384 features (3D keypoints) for letter classification. 

Hamerprocessing.py
Input a CSV file with the video ID and fingerspelling events recorded and outputs an h5 file with hand information. 
