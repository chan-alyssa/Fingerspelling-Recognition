# Recognising BSL Fingerspelling in Continuous Signing Sequences


## Enviroment
- ```pip install -r requirements.txt ```
- Create conda environment for fs23k ``` conda create --name fs23k python=3.10
conda activate fs23k ```

 ## Hand Feature Extraction

- Clone the HaMer repository
- Download requirements
- Input a video into processingpaper.py to extract pose and shape parameters and save in h5 file
- Extract the 384 hand features (for letter recognition) and 86 hand features (for fingerspelling detection) using extractfeatures.py which add the features into the existing h5 file

## Lip Feature Extraction
- Clone the AUTO-AVSR repository
- Download requirements
- Extract the lip features using extractautoavsr.py

## Inference
- Download trained models and scalars for detection and letter recognition
- Extractfeatures.py already has 'cleanedlabel', 1 if fingerspelling is present in the frame and 0 otherwise
- Use extractctcword.py to show predicted word
- Use display.py and makevideo.py to show each from images saved to --out_folder and make a video
 

