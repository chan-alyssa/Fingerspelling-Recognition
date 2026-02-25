import cv2
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description='Sliding Window IoU Detection')
    # parser.add_argument('--hfile_dir', type=str, help='Path to the directory containing left.npy and right.npy files')
    # parser.add_argument('--vid', type=str, help='Path to the video file')
    # parser.add_argument('--csv_file', type=str, help='Path to the CSV file containing annotations')
    # parser.add_argument('--startframe', type=int, help='Start frame number for analysis (0-indexed)')
    # parser.add_argument('--endframe', type=int, help='End frame number for analysis (inclusive)')
    # parser.add_argument('--threshold', type = float)
    parser.add_argument('--frame_dir', type = str)
    parser.add_argument('--output_video', type=str)
    #parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT, help='Path to pretrained model checkpoint')
    parser.add_argument('--frame_rate', type=str)
    args = parser.parse_args()

    frame_dir = args.frame_dir
    output_video_path = args.output_video
    frame_rate = int(args.frame_rate)

    #if ends with png sort
    frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith(".jpg")])

    #get the dimensions
    first_frame = cv2.imread(os.path.join(frame_dir, frame_files[0]))
    height, width, layers = first_frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format
    video_writer = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))

    for frame_file in frame_files:
        frame_path = os.path.join(frame_dir, frame_file)
        frame = cv2.imread(frame_path)  # Read the frame
        video_writer.write(frame)       # Write the frame to the video

    video_writer.release()
    print(f"Video saved to {output_video_path}")

if __name__ == "__main__":
    main()
