#if merging first and then extract features
#getting joint information
from pathlib import Path
import numpy as np
import argparse
import cv2
import os.path as osp
import h5py
import tarfile
import re
import os
from hamer.models import MANO
from hamer.utils.renderer import Renderer, cam_crop_to_full
from readh5 import ReadH5
from PIL import Image
import torch
import pandas as pd
import matplotlib.pyplot as plt
import math
import io
import torch.nn as nn

import joblib
LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)
LIGHT_BLUE = (LIGHT_BLUE[2], LIGHT_BLUE[1], LIGHT_BLUE[0])

class BinaryClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=128, dropout_prob=0.2):
        super(BinaryClassifier, self).__init__()
        
        # First hidden layer
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.layer_norm1 = nn.LayerNorm(hidden_size)  
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_prob)
        
        # Second hidden layer (newly added)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.layer_norm2 = nn.LayerNorm(hidden_size // 2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_prob)

        # Third hidden layer (newly added)
        self.fc3 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.layer_norm3 = nn.LayerNorm(hidden_size // 4)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout_prob)
        
        # Output layer
        self.fc4 = nn.Linear(hidden_size // 4, 1)  

    def forward(self, x):
        x = self.fc1(x)
        x = self.layer_norm1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.layer_norm2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        x = self.layer_norm3(x)
        x = self.relu3(x)
        x = self.dropout3(x)

        x = self.fc4(x)  # Output layer (no activation inside the model)
        return x
    
# Define the linear classifier
class LinearClassifier(nn.Module):
    def __init__(self, input_size):
        super(LinearClassifier, self).__init__()
        self.linear = nn.Linear(input_size, 1)  # Linear layer with a single output

    def forward(self, x):
        return self.linear(x)
    
def linear_classifier(features,weights,bias):
    score = np.dot(weights,features) + bias
    return 1 if score > 0 else 0

def check_fingertip_touch(index_tip, other_tips, threshold=5):
    for tip in other_tips:
        if np.linalg.norm(index_tip - tip) < threshold:
            return True
    return False

def project_hand_pose(hand_pose, camera_matrix, pred_cam_right):
    # Project 3D hand pose to 2D using the camera matrix
    default_camera_matrix = np.array([[1, 0, 0, pred_cam_right[0]], [0, 1, 0,pred_cam_right[1]], [0, 0, 1,pred_cam_right[2]]])
    hand_pose_homogeneous = np.hstack((hand_pose, np.ones((hand_pose.shape[0], 1))))
    # print("hand_pose_homogeneous",np.shape(hand_pose_homogeneous))
    # print("camera_matrix",np.shape(camera_matrix))
    projected_pose = camera_matrix @ default_camera_matrix@ hand_pose_homogeneous.T
    projected_pose /= projected_pose[2, :]
    return projected_pose[:2, :].T

def calculate_fingertip_centre(fingertips):
    fingertips_array = np.array(fingertips)
    if np.all(fingertips_array ==0):
        return None
    centre = np.mean(fingertips_array, axis=0)
    return centre


def distance(a, b):
    if len(a) == 2 and len(b) == 2:
        return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
    elif len(a) == 3 and len(b) == 3:
        return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)



def calculate_metrics(predictions, ground_truth):
    """Calculate evaluation metrics."""
    TP = np.sum((predictions == 1) & (ground_truth == 1))
    TN = np.sum((predictions == 0) & (ground_truth == 0))
    FP = np.sum((predictions == 1) & (ground_truth == 0))
    FN = np.sum((predictions == 0) & (ground_truth == 1))

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (TP + TN) / len(ground_truth)

    return precision, recall, f1_score, accuracy


def calculate_iou(predictions, ground_truth):
    """Calculate Intersection over Union (IoU)."""
    intersection = np.sum((predictions == 1) & (ground_truth == 1))
    union = np.sum((predictions == 1) | (ground_truth == 1))
    return intersection / union if union > 0 else 0


def sliding_window_fingerspelling(predictions, ground_truth, frame_rate):
    """Apply sliding window to check for fingerspelling based on IoU."""
    window_size = int(frame_rate)  # 1-second window
    step_size = 1
    num_frames = len(predictions)

    is_fingerspelling_values = np.zeros(num_frames)
    
    for start in range(0, num_frames - window_size + 1, step_size):
        end = start + window_size
        window_predictions = predictions[start:end]
        window_ground_truth = ground_truth[start:end]
        
        # Calculate IoU for the window
        iou = calculate_iou(window_predictions, window_ground_truth)
        is_fingerspelling = 1 if iou > 0.5 else 0
        if is_fingerspelling == 1:
            is_fingerspelling_values[start:end] = 1

    return is_fingerspelling_values

def fingerspelling(predictions, window_size, threshold):
    step_size=1
    num_frames = len(predictions)

    fingerspelling_values = np.zeros(num_frames)

    for start in range(0, num_frames - window_size + 1, step_size):
        end = start + window_size
        window_predictions = predictions[start:end]

        # Check if the proportion of predicted frames > 0.5 in the window exceeds the threshold
        proportion_active = np.sum(window_predictions)
        fingerspelling = 1 if proportion_active >= threshold else 0

        if fingerspelling == 1:
            fingerspelling_values[start:end] = 1

    return fingerspelling_values
def main():

    parser = argparse.ArgumentParser(description='Sliding Window IoU Detection')
    # parser.add_argument('--directory', type=str, help='Path to the directory containing left.npy and right.npy files')
    # parser.add_argument('--in_folder', type=str, help='Path to the directory containing left.npy and right.npy files')
    parser.add_argument('--hfile', type=str, help='Path to the directory containing left.npy and right.npy files')
    args = parser.parse_args()
    
    mano_cfg = {'data_dir': '_DATA/data/', 'model_path':'_DATA/data/mano/',
                'gender':'neutral','num_hand_joints':15,'mean_params':'./_DATA/data/mano_mean_params.npz','create_body_pose': False}
    mano = MANO(**mano_cfg)

    model1 = BinaryClassifier(86)
    model1.load_state_dict(torch.load('binary_classifierf86.pth'))
    model1.eval()

    scaler = joblib.load('scalerf86.pkl')
    # directory = args.directory
    # for filename in os.listdir(directory):

    #     hfile_path = os.path.join(directory, f"{filename}")
    #     print(hfile_path)
    h5file_path = args.hfile
    if h5file_path.endswith('.h5'):
    # for h5file_name in os.listdir(args.in_folder):
    #     if h5file_name.endswith('.h5'):
    #         h5file_path = os.path.join(args.in_folder, h5file_name)
        read_h5file = ReadH5(h5file_path)
        f = h5py.File(h5file_path, 'a')
        video_grp = f['video']


        # read_h5file = ReadH5(f'{args.hfile}.h5')
        # f = h5py.File(f'{args.hfile}.h5')
        dataset = f['video/hand_pose']
        total_frames = dataset.shape[0]
        # with h5py.File(h5file_path, 'a') as f:
        #     video_grp = f['video']

        # right = np.zeros((total_frames,2))
        # left = np.zeros((total_frames,2))
        cnt = 0
            # hf = h5py.File(hfile_path, 'a')
            # g1 = hf.create_group('more')
            # g1.create_dataset('features',(total_frames,5))
            # data1=['more/features']
        del f['video/features86']
        del f['video/features384']
        del f['video/label']
        del f['video/cleanedlabel']
        del f['video/frame_number']
        video_grp.create_dataset('features86', (total_frames, 86))
        data1 = video_grp['features86']
        video_grp.create_dataset('features384', (total_frames, 384))
        data2 = video_grp['features384']
        video_grp.create_dataset('label', (total_frames, 1))
        data10 = video_grp['label']
        video_grp.create_dataset('cleanedlabel', (total_frames, 1))
        data9 = video_grp['cleanedlabel']
        # video_grp.create_dataset('video_name', (total_frames, 1), dtype='i8') 
        # data4 = video_grp['video_name']
        video_grp.create_dataset('frame_number', (total_frames, 1))
        data7 = video_grp['frame_number']
        str_dtype = h5py.string_dtype(encoding='utf-8')
        # video_grp.create_dataset('subtitle', (total_frames,), dtype=str_dtype)
        # data8 = video_grp['subtitle']
        predictions = np.zeros(total_frames)


        for cnt in range(total_frames):
            all_verts = []
            all_cam_t = []
            all_right = []   

            data = read_h5file.read_sequence_slice('video/hand_pose', cnt)
            #data3 = read_h5file.read_sequence_slice('video/label', cnt)
            # data5 = read_h5file.read_sequence_slice('video/video_name', cnt)
            # data8[cnt]= read_h5file.read_sequence_slice('video/subtitle', cnt)

            #data2[cnt] = data3
            # data4[cnt] = data5
            data7[cnt] = int(data[0])
            # frame_number = int(data[0])
            # cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            # ret, frame = cap.read()


            left_exist = int(data[1])
            right_exist = int(data[162])
            img_size = torch.tensor([[456., 256.]])
            FOCAL_LENGTH = 5000
            MODEL_IMG_SIZE = 224
            scaled_focal_length = FOCAL_LENGTH / MODEL_IMG_SIZE * img_size.max()

            
            camera_matrix = np.array([[scaled_focal_length , 0, img_size[0][0]/2],
                            [0, scaled_focal_length , img_size[0][1]/2],
                            [0, 0, 1]])
            if left_exist:
                arr_left = data[2:162]
                box_center_left = torch.tensor([arr_left[:2]])
                box_size_left = torch.tensor([arr_left[2]])
                pred_cam_left = torch.tensor([arr_left[3:6]])
                global_orient_left = torch.tensor([np.reshape(arr_left[6:15], (1, 3, 3))])
                hand_pose_left = torch.tensor([np.reshape(arr_left[15:150], (15, 3, 3))])
                betas_left = torch.tensor([arr_left[150:]])
                is_right=0
                mano_output_left = mano(**{'global_orient': global_orient_left, 'hand_pose': hand_pose_left, 'betas': betas_left}, pose2rot=False)
                pred_keypoints_3d_left = mano_output_left.joints.reshape(-1, 3).numpy()
                pred_keypoints_3d_left[:,0] = (2*is_right-1)*pred_keypoints_3d_left[:,0]
                pred_cam_t_full_left = cam_crop_to_full(pred_cam_left, box_center_left, box_size_left, img_size, scaled_focal_length).detach().cpu().numpy()
                projected_pose_left = project_hand_pose(pred_keypoints_3d_left, camera_matrix, pred_cam_t_full_left[0])
                feature_left = projected_pose_left.flatten()
                feature_l = pred_keypoints_3d_left+pred_cam_t_full_left
                feature_left1 = feature_l.flatten() 
                fingertips_left = [projected_pose_left[4], projected_pose_left[8],
                                        projected_pose_left[12], projected_pose_left[16], projected_pose_left[20]]
                left_centre = calculate_fingertip_centre(fingertips_left)
                fingertips_left_3d = [pred_keypoints_3d_left[4], pred_keypoints_3d_left[8],
                                        pred_keypoints_3d_left[12], pred_keypoints_3d_left[16], pred_keypoints_3d_left[20]]
                left_centre_3d = calculate_fingertip_centre(fingertips_left_3d)


            if right_exist:
                arr_right = data[163:]
                box_center_right = torch.tensor([arr_right[:2]])
                box_size_right = torch.tensor([arr_right[2]])
                pred_cam_right = torch.tensor([arr_right[3:6]])
                global_orient_right = torch.tensor([np.reshape(arr_right[6:15], (1, 3, 3))])
                hand_pose_right = torch.tensor([np.reshape(arr_right[15:150], (15, 3, 3))])
                betas_right = torch.tensor([arr_right[150:]])

                mano_output_right = mano(**{'global_orient': global_orient_right, 'hand_pose': hand_pose_right, 'betas': betas_right}, pose2rot=False)
                pred_keypoints_3d_right = mano_output_right.joints.reshape(-1, 3).numpy()
                pred_cam_t_full_right = cam_crop_to_full(pred_cam_right, box_center_right, box_size_right, img_size, scaled_focal_length).detach().cpu().numpy()
                projected_pose_right = project_hand_pose(pred_keypoints_3d_right, camera_matrix, pred_cam_t_full_right[0])
                feature_right = projected_pose_right.flatten() 
                feature_r = pred_keypoints_3d_right+pred_cam_t_full_right
                feature_right1 = feature_r.flatten() 
                fingertips_right = [projected_pose_right[4], projected_pose_right[8],
                                            projected_pose_right[12], projected_pose_right[16], projected_pose_right[20]]
                right_centre = calculate_fingertip_centre(fingertips_right)
                fingertips_right_3d = [pred_keypoints_3d_right[4], pred_keypoints_3d_right[8],
                                            pred_keypoints_3d_right[12], pred_keypoints_3d_right[16], pred_keypoints_3d_right[20]]
                right_centre_3d = calculate_fingertip_centre(fingertips_right_3d)
            if right_exist and left_exist:

                # if cnt > 1:
                #     right_move = distance(right[cnt], right[cnt-1])
                #     left_move = distance(left[cnt], left[cnt - 1])
                # else:
                #     right_move = 0
                #     left_move = 0
                # print(feature_left.shape)

                features = np.concatenate([feature_right, feature_left,  np.array([distance(right_centre,left_centre)]),np.array([distance(right_centre_3d, left_centre_3d)])])
                if cnt == 0:
                    prev_frame=np.zeros(128)
                else:
                    prev_frame = data2[cnt-1,:128]
                if cnt == 1 or cnt ==0:
                    prev_prev_frame=np.zeros(128)
                else:
                    prev_prev_frame = data2[cnt-2,:128]
                                
                features1 = np.concatenate([feature_right1, feature_left1,  np.array([distance(right_centre,left_centre)]),np.array([distance(right_centre_3d, left_centre_3d)]),prev_frame, prev_prev_frame])
            else:
                features = np.zeros(feature_right.shape[0]+ feature_left.shape[0]+2)
                features1 = np.zeros(384)
            data1[cnt] = features
            data2[cnt] = features1

            features=np.array(features)
            features = features.reshape(1, -1)
            features = scaler.transform(features)
            features=torch.FloatTensor(features)

            with torch.no_grad():  # No need to compute gradients during inference
                prediction = model1(features)
                probability = torch.sigmoid(prediction)
                predicted_class = (probability > 0.5).float()
                # print(probability)
                # data9[cnt]= predicted_class
                predictions[cnt]= predicted_class
                data10[cnt]= predicted_class
            # print(cnt)

            cnt+=1

        results = fingerspelling(predictions, 8,6)

        for ii in range(0,total_frames):
            data9[ii]=results[ii]






if __name__ == '__main__':
    main()
