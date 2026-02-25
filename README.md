# Recognising BSL Fingerspelling in Continuous Signing Sequences


## Enviroment and Checkpoints
- ```pip install -r requirements.txt ```
- Create conda environment for fs23k ``` conda create --name fs23k python=3.10
conda activate fs23k ```


 ## Hand Feature Extraction

- Clone the HaMer repository
- Download requirements
- Input a video into processingpaper.py to extract pose and shape parameters and save in h5 file
- Extract the 384 hand features (for letter recognition) and 86 hand features (for fingerspelling detection) using extracthandfeatures.py which add the features into the existing h5 file

## Lip Feature Extraction
- Clone the AUTO-AVSR repository
- Download requirements
- Extract the lip features using ..

## Inference
- Download trained models and scalars for detection and letter recognition
- Use prediction86paper.py to show fingerspelling intervals and producing per frame images with a graph showing predictions per frame and smoothed predictions. These images are saved to a folder --out_folder
- Use predictions... to show prediction of the predicted word and frame-level letter annotations. Images saved to --out_folder.
 
## Reproducing the scores on the test set
``` python test.py --ckpt_path MODEL.pth ... add more ```
