# sliding window
# trying to create the graphs after
#want to iterate through h files in a folder
#also included the iou metrics from binary 9 but haven't tested because CUDA out of memory :(
from pathlib import Path
import numpy as np
import argparse
import cv2
import os.path as osp
import h5py
import tarfile
import re
import os
from readh5 import ReadH5
from PIL import Image
import torch
import pandas as pd
import matplotlib.pyplot as plt
import math
import io
# from hamer.models import MANO
# from hamer.configs import CACHE_DIR_HAMER
# from hamer.models import HAMER, download_models, load_hamer, DEFAULT_CHECKPOINT
# from hamer.utils import recursive_to
# from hamer.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
# from hamer.utils.renderer import Renderer, cam_crop_to_full
# from vitpose_model import ViTPoseModel
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import joblib

LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)
LIGHT_BLUE = (LIGHT_BLUE[2], LIGHT_BLUE[1], LIGHT_BLUE[0])
COLOUR=(0.2399,  0.5698,  0.8227)


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


def calculate_iou(predictions, ground_truth):
    """Calculate Intersection over Union (IoU)."""
    intersection = np.sum((predictions == 1) & (ground_truth == 1))
    union = np.sum((predictions == 1) | (ground_truth == 1))
    return intersection / union if union > 0 else 0


def calculate_iou_and_metrics(predictions, ground_truth, video_list, iou_threshold=0.5):
    tp = 0
    fp = 0

    ground_event = False  # Track if we are inside a ground truth event
    pred_event = False  # Track if we are inside a predicted event

    intersection = 0
    union = 0
    total_gt_events = 0
    total_pred_events = 0
    event_matrix = np.zeros(len(predictions))  # Initialize matrix with zeros
    tp_event = np.zeros(len(predictions))
    gt_event = np.zeros(len(predictions))
    p_event = np.zeros(len(predictions))
    fn_event = np.zeros(len(predictions))
    fp_event = np.zeros(len(predictions))

    previous_video = None  # Track the previous video to detect changes

    for i in range(len(predictions)):
        pred = predictions[i]
        gt = ground_truth[i]
        video_name = video_list[i]

        # Reset IoU tracking if the video has changed
        if video_name != previous_video:
            intersection = 0
            union = 0
            ground_event = False
            pred_event = False

        previous_video = video_name  # Update previous video tracker

        # Detect start of a new ground truth event
        if gt == 1 and not ground_event:
            ground_event = True
            total_gt_events += 1  # Count only when an event starts

        if pred == 1 and not pred_event:
            pred_event = True
            total_pred_events += 1

        # Track intersection and union
        if ground_event or pred_event:
            union += 1
        if ground_event and pred_event:
            intersection += 1

        if pred == 0 and pred_event and not ground_event:
            if intersection/union < iou_threshold:
                fp += 1 
                pred_event = False
                intersection = 0
                union = 0

        # Check if both events ended (a complete event window)
        if (pred == 0 and gt == 0) and (pred_event or ground_event):
            iou = intersection / union if union > 0 else 0
            if iou >= iou_threshold:
                tp += 1 
                event_matrix[i - union: i + 1] = 1
                tp_event[i - union:] = tp 
            pred_event = False
            ground_event = False
            intersection = 0
            union = 0

        if gt == 0 and ground_event:
            ground_event = False  # Reset when the ground truth event ends
        
        if pred == 0 and pred_event:
            pred_event = False 

        gt_event[i] = total_gt_events
        p_event[i] = total_pred_events
    fn_event = gt_event - tp_event
    fp_event = p_event - tp_event
    true_pos = tp_event[-1]
    false_neg = fn_event[-1]
    false_pos = fp_event[-1]
    return true_pos, false_neg, false_pos, event_matrix, tp_event, fn_event, fp_event


def smooth(predictions):
    results = np.zeros(len(predictions))
    for ii in range(2,len(predictions)-2):
        if predictions[ii-1] ==1 and predictions[ii+1]==1 and predictions[ii]==1:
            results[ii] = 1 
    return results

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


def main():
    parser = argparse.ArgumentParser(description='Sliding Window IoU Detection')
    # parser.add_argument('--hfile_dir', type=str, help='Path to the directory containing left.npy and right.npy files')
    # parser.add_argument('--vid', type=str, help='Path to the video file')
    # parser.add_argument('--csv_file', type=str, help='Path to the CSV file containing annotations')
    # parser.add_argument('--startframe', type=int, help='Start frame number for analysis (0-indexed)')
    # parser.add_argument('--endframe', type=int, help='End frame number for analysis (inclusive)')
    # parser.add_argument('--threshold', type = float)
    parser.add_argument('--out_folder', type = str)
    parser.add_argument('--h5file', type=str)
    #parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT, help='Path to pretrained model checkpoint')
    parser.add_argument('--vid', type=str)
    args = parser.parse_args()

    
    # model, model_cfg = load_hamer(args.checkpoint)
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # model = model.to(device)
    # model.eval()
    model1 = BinaryClassifier(86)
    model1.load_state_dict(torch.load('binary_classifierf86.pth'))
    model1.eval()



    # renderer = Renderer(model_cfg, faces=model.mano.faces)
    
    # mano_cfg = {'data_dir': '_DATA/data/', 'model_path':'_DATA/data/mano/',
    #             'gender':'neutral','num_hand_joints':15,'mean_params':'./_DATA/data/mano_mean_params.npz','create_body_pose': False}
    # mano = MANO(**mano_cfg)

    scaler = joblib.load('scalerf86.pkl')
    #try to iterate through folder of h5 files and do this
    h5file_path = args.h5file
    video_path = args.vid
    if h5file_path.endswith('.h5'):
        read_h5file = ReadH5(h5file_path)
        f = h5py.File(h5file_path)
        file_without_ext = os.path.splitext(h5file_path)[0]
        dataset = f['video/hand_pose']
        total_frames = dataset.shape[0]-1

        out_folder = args.out_folder
        os.makedirs(out_folder, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        subtitle = np.empty(num_frames, dtype=object)
        
        # Get video properties
        frame_rate = cap.get(cv2.CAP_PROP_FPS)

        # Initialize the ground truth array
        ground_truth = np.zeros(num_frames)
        predictions = np.zeros(num_frames)
        right = np.zeros((num_frames,2))
        left = np.zeros((num_frames,2))
        video_list= np.zeros(num_frames)
        results = np.zeros(num_frames)
        wordpredictions = np.empty(num_frames, dtype=object)

        for cnt in range(0,total_frames):

            cleanedlabel =  read_h5file.read_sequence_slice('video/cleanedlabel', cnt)
            frame_number = read_h5file.read_sequence_slice('video/frame_number', cnt)
            number = int(frame_number)
            # print(number)
            results[number] = cleanedlabel[0]


        for ii in range(0,total_frames):
            # if ii % 500 != 0:
            #     continue

                
            all_verts = []
            all_cam_t = []
            all_right = []   
            
            data = read_h5file.read_sequence_slice('video/hand_pose', ii)
            cleanedlabel =  read_h5file.read_sequence_slice('video/cleanedlabel', ii)
            framewiselabel = read_h5file.read_sequence_slice('video/label', ii)
            subtitle = read_h5file.read_sequence_slice('video/ctcword', ii)
            subtitle_text = subtitle.decode('utf-8')
            frame_number = int(data[0])
            label = cleanedlabel[0]
            wordpredictions[frame_number] = subtitle_text
            # if ground_truth[frame_number] != 1:
            #     continue
            # if frame_number < 17400:
            #     continue


            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
                
            # all_verts = []
            # all_cam_t = []
            # all_right = []   
            

            # if ret:
                
            #     left_exist = int(data[1])
            #     right_exist = int(data[162])
            #     # Set the image size and other parameters for the camera
            #     img_size = torch.tensor([[np.shape(frame)[1], np.shape(frame)[0]]]).float()
            #     FOCAL_LENGTH = 5000
            #     MODEL_IMG_SIZE = 224
            #     scaled_focal_length = FOCAL_LENGTH / MODEL_IMG_SIZE * img_size.max()

            #     camera_matrix = np.array([[scaled_focal_length , 0, np.shape(frame)[1]/2],
            #                     [0, scaled_focal_length , np.shape(frame)[0]/2],
            #                     [0, 0, 1]])
                
            #     if left_exist:
            #         #print(data)
            #         arr_left = data[2:162]

            #         # Extract parameters for the left hand
            #         box_center_left = torch.tensor([arr_left[:2]])
            #         box_size_left = torch.tensor([arr_left[2]])
            #         pred_cam_left = torch.tensor([arr_left[3:6]])
            #         global_orient_left = torch.tensor([np.reshape(arr_left[6:15], (1, 3, 3))])
            #         hand_pose_left = torch.tensor([np.reshape(arr_left[15:150], (15, 3, 3))])
            #         betas_left = torch.tensor([arr_left[150:]])

            #         # Left hand parameters to MANO model
            #         pred_mano_params_left = {
            #             'global_orient': global_orient_left,
            #             'hand_pose': hand_pose_left,
            #             'betas': betas_left
            #         }
            #         pred_mano_params_left['global_orient'] = pred_mano_params_left['global_orient'].reshape(1, -1, 3, 3)
            #         pred_mano_params_left['hand_pose'] = pred_mano_params_left['hand_pose'].reshape(1, -1, 3, 3)
            #         pred_mano_params_left['betas'] = pred_mano_params_left['betas'].reshape(1, -1)

            #         # Compute vertices for the left hand
            #         mano_output_left = mano(**{k: v.float() for k, v in pred_mano_params_left.items()}, pose2rot=False)
            #         pred_keypoints_3d_left = mano_output_left.joints
            #         pred_vertices_left = mano_output_left.vertices.numpy()
            #         pred_keypoints_3d_left = pred_keypoints_3d_left.reshape(-1, 3)
            #         pred_vertices_left = pred_vertices_left.reshape(-1, 3)
            #         pred_keypoints_3d_left[:, 0] = -pred_keypoints_3d_left[:, 0]

            #         # Project camera translation for the left hand
            #         pred_cam_t_full_left = cam_crop_to_full(pred_cam_left, box_center_left, box_size_left, img_size, scaled_focal_length).detach().cpu().numpy()

            #         # Process left hand
            #         verts_left = pred_vertices_left
            #         is_right = 0 # Left hand
            #         verts_left[:, 0] = (2 * is_right - 1) * verts_left[:, 0]  # Flip x-axis for left hand
            #         cam_t_left = pred_cam_t_full_left
            #         all_verts.append(verts_left)
            #         all_cam_t.append(cam_t_left)
            #         all_right.append(is_right)

                
            #     if right_exist:
            #         arr_right = data[163:]

            #         # Extract parameters for the right hand
            #         box_center_right = torch.tensor([arr_right[:2]])
            #         box_size_right = torch.tensor([arr_right[2]])
            #         pred_cam_right = torch.tensor([arr_right[3:6]])
            #         global_orient_right = torch.tensor([np.reshape(arr_right[6:15], (1, 3, 3))])
            #         hand_pose_right = torch.tensor([np.reshape(arr_right[15:150], (15, 3, 3))])
            #         betas_right = torch.tensor([arr_right[150:]])

            #         # Right hand parameters to MANO model
            #         pred_mano_params_right = {
            #             'global_orient': global_orient_right,
            #             'hand_pose': hand_pose_right,
            #             'betas': betas_right
            #         }
            #         pred_mano_params_right['global_orient'] = pred_mano_params_right['global_orient'].reshape(1, -1, 3, 3)
            #         pred_mano_params_right['hand_pose'] = pred_mano_params_right['hand_pose'].reshape(1, -1, 3, 3)
            #         pred_mano_params_right['betas'] = pred_mano_params_right['betas'].reshape(1, -1)

            #         # Compute vertices for the right hand
            #         mano_output_right = mano(**{k: v.float() for k, v in pred_mano_params_right.items()}, pose2rot=False)
            #         pred_keypoints_3d_right = mano_output_right.joints
            #         pred_vertices_right = mano_output_right.vertices.numpy()
            #         pred_keypoints_3d_right = pred_keypoints_3d_right.reshape(-1, 3)
            #         pred_vertices_right = pred_vertices_right.reshape(-1, 3)
            #         # Project camera translation for the right hand
            #         pred_cam_t_full_right = cam_crop_to_full(pred_cam_right, box_center_right, box_size_right, img_size, scaled_focal_length).detach().cpu().numpy()

            #         # Process right hand
            #         verts_right = pred_vertices_right
            #         is_right = 1  # Right hand
            #         verts_right[:, 0] = (2 * is_right - 1) * verts_right[:, 0]  # No flipping for right hand
            #         cam_t_right = pred_cam_t_full_right
            #         all_verts.append(verts_right)
            #         all_cam_t.append(cam_t_right)
            #         all_right.append(is_right)

            #     # if results[frame_number] == 1:
            #     mesh_base_color=LIGHT_BLUE
            #     #     mesh_base_color = COLOUR
            #     # else:



            #     # print("pred_cam_right",pred_cam_right)
            #     # print("all_cam_t",all_cam_t)
            #     # print("projected_pose",projected_pose)
            #     # print("img_size",img_size)
                
            #     misc_args = dict(
            #             mesh_base_color=mesh_base_color,
            #             scene_bg_color=(1, 1, 1),
            #             focal_length=scaled_focal_length,
            #             )
            #     cam_view = renderer.render_rgba_multiple(all_verts, cam_t=all_cam_t, render_res=img_size[0], is_right=all_right, **misc_args)

            # Overlay image
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # input_img = frame.astype(np.float32)[:,:,::-1]/255.0
            # input_img = np.concatenate([input_img, np.ones_like(input_img[:,:,:1])], axis=2) # Add alpha channel
            # input_img_overlay = input_img[:,:,:3] * (1-cam_view[:,:,3:]) + cam_view[:,:,:3] * cam_view[:,:,3:]
            # output_image = (255 * input_img_overlay[:, :, ::-1]).astype(np.uint8)
            window_size = 300  # Number of frames to display around the fixed frame

            # Calculate the start and end indices for the current window
            start_index = (frame_number // window_size) * window_size
            end_index = min(start_index + window_size, num_frames)

            # truep = tp[end_index] - tp[start_index]
            # falsen = fn[end_index] - fn[start_index]
            # falsep=fp[end_index]- fp[start_index]
            # event_type = event_matrix[frame_number]

            # truep1 = tp_1[end_index] - tp_1[start_index]
            # falsen1 = fn_1[end_index] - fn_1[start_index]
            # falsep1=fp_1[end_index]- fp_1[start_index]


            # # Set the color based on the event type
            # if event_type == 1:
            # #     line_colour = 'blue'  # TP (True Positive)
            # #     # print('BLUE')
            # # else:
            line_colour = 'black'  # Default color if no event type is set


            # print(total_tp, total_fn, total_fp)
            # Create the figure and subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [2, 1]})
            plt.subplots_adjust(hspace=0.3)
            # --- Top: Video Frame ---
            # Display the rendered frame (current frame visualization)
            ax1.imshow(frame)
            ax1.axis("off")  # Hide axis for better display
            ax1.set_title(f"Frame {frame_number}")

            # if current_subtitle!="richard":
            #     continue
            if label == 1:
              ax1.text(10, 30, f"Predicted word: {subtitle_text}", fontsize=15, color='white', bbox=dict(facecolor='black', alpha=0.7))

            # --- Bottom: Ground Truth vs Model Results Timeline ---
            # Plot the timeline with a fixed x-axis range
            #ax2.plot(range(num_frames), predictions, color='blue', lw=2, label="Smoothed Results")
            ax2.plot(range(num_frames), results, color='blue', lw=2, label="Smooth Results")
            ax2.axvline(frame_number, color=line_colour, linestyle='--', lw=1.5, label='Current Frame')

            # Set the x-axis limits to the current window range
            ax2.set_xlim(start_index, end_index)

            # Set labels and title for the timeline plot
            ax2.set_ylim(0, 1.5)  # Adjust y-axis limits as necessary
            # ax2.set_xlabel("Frame Number")
            ax2.set_ylabel("Predictions")
            ax2.set_title(f"Video {video_path}")


            frame_number2 = 2000+frame_number
            output_path = os.path.join(out_folder, f'{frame_number2}_all.jpg')
            plt.savefig(output_path)
            plt.close()


            


if __name__ == '__main__':
    main()  
