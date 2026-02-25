import argparse
import os
import sys
import glob
import torch
import h5py
import numpy as np
import re
import cv2
import torchvision
from lightning import ModelModule
from datamodule.transforms import AudioTransform, VideoTransform
from readh5 import ReadH5

sys.path.insert(0, "../")
import torchaudio




class InferencePipeline(torch.nn.Module):
    def __init__(self, args, ckpt_path, detector="retinaface"):
        super(InferencePipeline, self).__init__()
        self.modality = args.modality

        if self.modality == "audio":
            self.audio_transform = AudioTransform(subset="test")

        elif self.modality == "video":
            if detector == "mediapipe":
                from preparation.detectors.mediapipe.detector import LandmarksDetector
                from preparation.detectors.mediapipe.video_process import VideoProcess
                self.landmarks_detector = LandmarksDetector()
                self.video_process = VideoProcess(convert_gray=False)
            elif detector == "retinaface":
                from preparation.detectors.retinaface.detector import LandmarksDetector
                from preparation.detectors.retinaface.video_process import VideoProcess
                self.landmarks_detector = LandmarksDetector(device="cuda:0")
                self.video_process = VideoProcess(convert_gray=False)
            self.video_transform = VideoTransform(subset="test")

        ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        self.modelmodule = ModelModule(args)
        self.modelmodule.model.load_state_dict(ckpt)
        self.modelmodule.eval()

    def forward(self, input_data):
        if isinstance(input_data, str):
            input_data = os.path.abspath(input_data)
            assert os.path.isfile(input_data), f"{input_data} does not exist."

        if self.modality == "audio":
            if isinstance(input_data, str):
                audio, sample_rate = self.load_audio(input_data)
            else:
                raise ValueError("For audio, only file paths are supported currently.")

            audio = self.audio_process(audio, sample_rate)
            audio = audio.transpose(1, 0)
            audio = self.audio_transform(audio)
            with torch.no_grad():
                transcript = self.modelmodule(audio)
            return transcript
        elif self.modality == "video":

            if isinstance(input_data, str):
                video = self.load_video(input_data)
            else:
                if isinstance(input_data, list):
                    video = np.stack(input_data)
                elif isinstance(input_data, np.ndarray):
                    video = input_data
                else:
                    raise ValueError("Video input must be a path, list, or np.ndarray.")


            video_t = torch.tensor(video, dtype=torch.float32)


            video_t = video_t.permute(0, 3, 1, 2)


            try:
                video_t = self.video_transform(video_t)
            except Exception as e:
                print("Transform failed, returning zero features.")
                num_frames = video.shape[0]
                zero_feat = torch.zeros((num_frames, 768), dtype=torch.float32)
                return zero_feat, None

            try:
                with torch.no_grad():
                    feat = self.modelmodule.model.frontend(video_t.unsqueeze(0))
                    feat = self.modelmodule.model.proj_encoder(feat)
                    feat, _ = self.modelmodule.model.encoder(feat, None)
                    feat = feat.squeeze(0)
                    transcript = self.modelmodule(video_t)
                return feat, transcript

            except Exception as e:
                print(f"Failed to extract features: {e}")
                num_frames = video.shape[0]
                zero_feat = torch.zeros((num_frames, 768), dtype=torch.float32)
                return zero_feat, None

        # elif self.modality == "video":
        #     if isinstance(input_data, str):
        #         video = self.load_video(input_data)
        #     else:
        #         if isinstance(input_data, list):
        #             video = np.stack(input_data)
        #         elif isinstance(input_data, np.ndarray):
        #             video = input_data
        #         else:
        #             raise ValueError("Video input must be a path, list, or np.ndarray.")

        #     # Landmark detection and pre-processing
        #     landmarks = self.landmarks_detector(video)
        #     video = self.video_process(video, landmarks)
        #     video = torch.tensor(video)
        #     video = video.permute((0, 3, 1, 2))
        #     video = self.video_transform(video)

        #     # Extract features
        #     with torch.no_grad():
        #         features = self.modelmodule.model.frontend(video.unsqueeze(0))
        #         features = self.modelmodule.model.proj_encoder(features)
        #         features, _ = self.modelmodule.model.encoder(features, None)
        #         features = features.squeeze(0)
        #         transcript = self.modelmodule(video)

        #     return features, transcript

    def load_audio(self, data_filename):
        waveform, sample_rate = torchaudio.load(data_filename, normalize=True)
        return waveform, sample_rate

    def load_video(self, data_filename):
        return torchvision.io.read_video(data_filename, pts_unit="sec")[0].numpy()

    def audio_process(self, waveform, sample_rate, target_sample_rate=16000):
        if sample_rate != target_sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, sample_rate, target_sample_rate
            )
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        return waveform




def main():
    parser = argparse.ArgumentParser(description='Lip feature extraction with mouthing event batching')
    parser.add_argument('--hfile', type=str, help='Path to the directory containing left.npy and right.npy files')
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--vid_name', type=str)
    args = parser.parse_args()

    model_path = args.model_path
    setattr(args, 'modality', 'video')
    pipeline = InferencePipeline(args, model_path, detector="mediapipe")

    if args.hfile.endswith('.h5'):
        h5_path = args.hfile
        read_h5file = ReadH5(h5_path)
        f = h5py.File(h5_path, 'a')
        dataset = f['video/label']
        total_frames = dataset.shape[0]
        if 'video/autoavsr' in f:
          del f['video/autoavsr']

        f.create_dataset('video/autoavsr', (total_frames, 768), dtype='float32')

        cnt = 0
        prev_video = None
        prev_frame = None
        cap = None
        frames = []

        video_name = args.vid_name
        cap = cv2.VideoCapture(video_name)

        while cnt < total_frames:

            current_frame = int(read_h5file.read_sequence_slice('video/frame_number', cnt))

            # Read the frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = cap.read()
            if not ret:
                print(f"Missing frame {current_frame} in {video_name}")
                cnt += 1
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            cnt += 1


        if frames:
            features, _ = pipeline(frames)
            for i in range(len(frames)):
                f['video/autoavsr'][cnt - len(frames) + i] = features[i].cpu().numpy().astype(np.float32)

        if cap is not None:
            cap.release()

if __name__ == '__main__':
    main()
