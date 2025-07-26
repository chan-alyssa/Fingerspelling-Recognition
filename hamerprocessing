
from pathlib import Path
import os.path as osp
import h5py
import numpy as np
import argparse
import tarfile
import re
import os
import cv2
from hamer.utils.renderer import Renderer, cam_crop_to_full
from library.readh5 import ReadH5
from PIL import Image
import torch
import pandas as pd
import matplotlib.pyplot as plt
import math
import io
from hamer.models import MANO
from hamer.configs import CACHE_DIR_HAMER
from hamer.models import HAMER, download_models, load_hamer, DEFAULT_CHECKPOINT
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hamer.utils.renderer import Renderer, cam_crop_to_full
from vitpose_model import ViTPoseModel
import tqdm




LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)
# LIGHT_BLUE = (LIGHT_BLUE[2], LIGHT_BLUE[1], LIGHT_BLUE[0])


def main():
    parser = argparse.ArgumentParser(description='HaMeR demo code')
    parser.add_argument('--vid', type=str, help='Video ID to run inference on')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT, help='Path to pretrained model checkpoint')
    parser.add_argument('--img_folder', type=str, default='demo2', help='Folder with input images')
    parser.add_argument('--out_folder', type=str, default='demoout2', help='Output folder to save rendered results')
    parser.add_argument('--side_view', dest='side_view', action='store_true', default=False, help='If set, render side view also')
    parser.add_argument('--full_frame', dest='full_frame', action='store_true', default=True, help='If set, render all people together also')
    parser.add_argument('--save_mesh', dest='save_mesh', action='store_true', default=False, help='If set, save meshes to disk also')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference/fitting')
    parser.add_argument('--rescale_factor', type=float, default=2.0, help='Factor for padding the bbox')
    parser.add_argument('--body_detector', type=str, default='regnety', choices=['vitdet', 'regnety'], help='Using regnety improves runtime and reduces memory')
    parser.add_argument('--file_type', nargs='+', default=['*.jpg', '*.png'], help='List of file extensions to consider')
    parser.add_argument('--viz', dest='viz', action='store_true', default=False, help='If set, render side view also')
    parser.add_argument('--vid_write', dest='vid_write', action='store_true', default=False, help='If set, vid_write')
    parser.add_argument('--csv_file', type = str)
    # parser.add_argument('--video_directory', type = str)
    args = parser.parse_args()

    # Setup the renderer
    # model_cfg ={}
    # model_cfg.EXTRA.FOCAL_LENGTH = 5000
    # model_cfg.MODEL.IMAGE_SIZE = 224
    # model_cfg.MODEL.IMAGE_STD = [0.229, 0.224, 0.225]
    # model_cfg.MODEL.IMAGE_MEAN = [0.485, 0.456, 0.406]
    model, model_cfg = load_hamer(args.checkpoint)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()
    renderer = Renderer(model_cfg, faces=model.mano.faces)
    print(device)

    frametesting = pd.read_csv(args.csv_file)

 

    mano_cfg = {'data_dir': '_DATA/data/', 'model_path':'_DATA/data/mano/',
                'gender':'neutral','num_hand_joints':15,'mean_params':'./_DATA/data/mano_mean_params.npz','create_body_pose': False}
    mano = MANO(**mano_cfg)

    from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
    if args.body_detector == 'vitdet':
        from detectron2.config import LazyConfig
        import hamer
        cfg_path = Path(hamer.__file__).parent/'configs'/'cascade_mask_rcnn_vitdet_h_75ep.py'
        detectron2_cfg = LazyConfig.load(str(cfg_path))
        detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
        detector = DefaultPredictor_Lazy(detectron2_cfg)
    elif args.body_detector == 'regnety':
        from detectron2 import model_zoo
        from detectron2.config import get_cfg
        detectron2_cfg = model_zoo.get_config('new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py', trained=True)
        detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
        detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh   = 0.4
        detector       = DefaultPredictor_Lazy(detectron2_cfg)

    cpm = ViTPoseModel(device)

    # # Make output directory if it does not exist
    # os.makedirs(args.out_folder, exist_ok=True)

    
    # hf[f'{vid}/hand_pose'] = data1
    # print("data1",data1[3])
    if not os.path.exists(args.out_folder):
        os.makedirs(args.out_folder)
        print('made')


    unique_videos = frametesting['video_id'].unique()

    for video_name in unique_videos:
        video_path = ('/work/alyssa/videos/'f'{video_name}.mp4')
        print(video_path)
        cap = cv2.VideoCapture(video_path)
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        video_annotations = frametesting[frametesting['video_id'] == video_name]


        for i, row in video_annotations.iterrows():
            print(f"Processing video: {video_name}, row: {i}")
            start = row['start']
            frame_start = math.floor(start*frame_rate)
            frame_number=frame_start
            end = row['end']
            frame_end = math.ceil(end*frame_rate)
            frame_len=frame_end-frame_start+1
            #80721
            h5_file_path = os.path.join(args.out_folder, f'{i}_{video_name}.h5')
            
            hf = h5py.File(h5_file_path, 'w')
            g1 = hf.create_group('video')
            d1 = np.zeros((frame_len,316 + 7))
            str_dtype = h5py.string_dtype(encoding='utf-8')
            g1.create_dataset('hand_pose', (frame_len, 316 + 7))
            g1.create_dataset('video_name', (frame_len, 1), dtype='i8') 
            g1.create_dataset('label', (frame_len, 1))
            g1.create_dataset('subtitle', (frame_len,), dtype=str_dtype)
            data1 = hf['video/hand_pose']
            data2 = hf['video/video_name']
            data3 = hf['video/label']
            data4 = hf['video/subtitle']
            cnt = 0


            while frame_number <= frame_end:
                if cnt >= frame_len:
                    print(f"Warning: cnt ({cnt}) exceeded frame_len ({frame_len})")
                    break  # Prevent out-of-bound access
                data3[cnt] = 1
                data4[cnt] = str(row['word'])

                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                if not cap.isOpened():  # Check if the video was successfully opened
                    print(f"Can't open: {video_name}")
                    
                data2[cnt]=int(video_name)
                # print(data2[cnt])
                ret, frame = cap.read()
                if not ret:
                    print((f"Warning:Frame {frame_number} not read"))
                if ret:
                    data1[cnt,0] = frame_number
                    det_out = detector(frame)
                    det_instances = det_out['instances']
                    valid_idx = (det_instances.pred_classes==0) & (det_instances.scores > 0.5)
                    pred_bboxes=det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
                    pred_scores=det_instances.scores[valid_idx].cpu().numpy()

                    # print("pred_bboxes",pred_bboxes)

                    # Detect human keypoints for each person
                    vitposes_out = cpm.predict_pose(
                        frame,
                        [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
                    )

                    bboxes = []
                    is_right = []

                    # print("vitposes_out",vitposes_out)
                    # print("pred_bboxes",np.shape(pred_bboxes))
                    # print("vitposes_out",np.shape(vitposes_out))
                    # Use hands based on hand keypoint detections
                # Use hands based on hand keypoint detections
                    for vitposes in vitposes_out:
                        left_hand_keyp = vitposes['keypoints'][-42:-21]
                        right_hand_keyp = vitposes['keypoints'][-21:]

                        # Rejecting not confident detections
                        keyp = left_hand_keyp
                        valid = keyp[:,2] > 0.5
                        if sum(valid) > 3:
                            bbox = [keyp[valid,0].min(), keyp[valid,1].min(), keyp[valid,0].max(), keyp[valid,1].max()]
                            bboxes.append(bbox)
                            is_right.append(0)
                        keyp = right_hand_keyp
                        valid = keyp[:,2] > 0.5
                        if sum(valid) > 3:
                            bbox = [keyp[valid,0].min(), keyp[valid,1].min(), keyp[valid,0].max(), keyp[valid,1].max()]
                            bboxes.append(bbox)
                            is_right.append(1)

                    if len(bboxes) == 0:
                        data1[cnt, :] = 0  
                        data1[cnt, 0] = frame_number 
                        data3[cnt] = 0 
                        data2[cnt] = int(video_name)
                        cnt+=1
                        frame_number+=1

                        continue

                    boxes = np.stack(bboxes)
                    right = np.stack(is_right)


                    # Run reconstruction on all detected hands
                    dataset = ViTDetDataset(model_cfg, frame, boxes, right, rescale_factor=args.rescale_factor)
                    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

                    all_verts = []
                    all_cam_t = []
                    all_right = []
                    
                    for batch in dataloader:
                        batch = recursive_to(batch, device)
                        with torch.no_grad():
                            out = model(batch)

                        # print("box_center",np.shape(box_center)) # 1,2
                        # print("box_size",np.shape(box_size)) # 1
                        multiplier = (2*batch['right']-1)
                        pred_cam = out['pred_cam']
                        pred_cam[:,1] = multiplier*pred_cam[:,1]
                        box_center = batch["box_center"].float()
                        box_size = batch["box_size"].float()
                        img_size = batch["img_size"].float()
                        multiplier = (2*batch['right']-1)
                        scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
                        # pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()

                        pred_cam_cpu = out['pred_cam'].cpu()
                        pred_mano_params = out['pred_mano_params']
                        global_orient = pred_mano_params['global_orient'].cpu()
                        hand_pose = pred_mano_params['hand_pose'].cpu()
                        betas = pred_mano_params['betas'].cpu()

                        # Render the result
                        batch_size = batch['img'].shape[0]

                        if batch_size > 1:  # If more than one hand is predicted, handle the first one
                            for n in range(2):
                                # Assuming frame_int is the correct frame number
                                if batch['right'][n]:  # Right hand detected
                                    data1[cnt, 158+4] = 1  # Mark right hand in the correct column
                                    right_data = np.concatenate((
                                        box_center[n].cpu().numpy().flatten(),
                                        box_size[n].cpu().numpy().flatten(),
                                        pred_cam_cpu[n].numpy().flatten(),
                                        global_orient[n].numpy().flatten(),
                                        hand_pose[n].numpy().flatten(),
                                        betas[n].numpy().flatten()), axis=0)
                                    data1[cnt, 159+4:] = right_data  # Store right hand data in the correct column

                                if not batch['right'][n]:  # Left hand detected
                                    data1[cnt, 1] = 1  # Mark left hand in the correct column
                                    left_data = np.concatenate((
                                        box_center[n].cpu().numpy().flatten(),
                                        box_size[n].cpu().numpy().flatten(),
                                        pred_cam_cpu[n].numpy().flatten(),
                                        global_orient[n].numpy().flatten(),
                                        hand_pose[n].numpy().flatten(),
                                        betas[n].numpy().flatten()), axis=0)
                                    data1[cnt, 2:158+4] = left_data  # Store left hand data in the correct column
                        else:
                            if batch['right']:  # Right hand detected
                                data1[cnt, 158+4] = 1  # Mark right hand in the correct column
                                right_data = np.concatenate((
                                    box_center.cpu().numpy().flatten(),
                                    box_size.cpu().numpy().flatten(),
                                    pred_cam_cpu.numpy().flatten(),
                                    global_orient.numpy().flatten(),
                                    hand_pose.numpy().flatten(),
                                    betas.numpy().flatten()), axis=0)
                                data1[cnt, 159+4:] = right_data  # Store right hand data in the correct column
                            if not batch['right']:  # Left hand detected
                                data1[cnt, 1] = 1  # Mark left hand in the correct column
                                left_data = np.concatenate((
                                    box_center.cpu().numpy().flatten(),
                                    box_size.cpu().numpy().flatten(),
                                    pred_cam_cpu.numpy().flatten(),
                                    global_orient.numpy().flatten(),
                                    hand_pose.numpy().flatten(),
                                    betas.numpy().flatten()), axis=0)
                                data1[cnt, 2:158+4] = left_data

                # print(frame_number)


                frame_number+=1
                cnt +=1


        
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
