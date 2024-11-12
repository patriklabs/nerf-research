import cv2
import os
import argparse


def extract_frames(video_path, output_folder, frame_rate=1):
    """
    Extracts frames from a video and saves them as images.

    Parameters:
    - video_path: Path to the video file.
    - output_folder: Folder where extracted frames will be saved.
    - frame_rate: Number of frames per second to extract. Default is 1.
    """
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get the total number of frames and the FPS of the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(fps / frame_rate)  # Interval for capturing frames

    print(
        f"Total frames: {total_frames}, FPS: {fps}, Capturing every {interval} frames."
    )

    frame_count = 0
    saved_count = 0

    # Loop through the frames of the video
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # Save frame as image every interval frames
        if frame_count % interval == 0:
            output_path = os.path.join(output_folder, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(output_path, frame)
            print(f"Saved frame {saved_count} to {output_path}")
            saved_count += 1

        frame_count += 1

    # Release the video capture object
    cap.release()
    print("Frame extraction completed.")


def main(video_path, output_folder, frame_rate):
    extract_frames(video_path, output_folder, frame_rate)


def add_arguments():
    parser = argparse.ArgumentParser(description="Extract frames from a video.")

    parser.add_argument("--video_path", type=str)
    parser.add_argument("--output_folder", type=str)
    parser.add_argument("--frame_rate", type=int, default=1)

    return parser.parse_args()


if __name__ == "__main__":

    args = add_arguments()
    main(args.video_path, args.output_folder, args.frame_rate)
