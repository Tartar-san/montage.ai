import scenedetect
import time
import cv2
import os

detector_list = [
        scenedetect.detectors.ThresholdDetector(threshold=16, min_percent=0.9),
        scenedetect.detectors.ContentDetector(min_scene_len=15)
    ]


def extract_chunks(video_path, chunks_path="chunks"):
    scene_list = []

    video_fps, frames_read = scenedetect.detect_scenes_file(
        video_path, scene_list, detector_list)

    scene_list.sort()

    video = cv2.VideoCapture(video_path)
    ret, frame = video.read()
    frame_number = 0
    chunk_number = 0

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    chunk_to_write = cv2.VideoWriter(
        os.path.join(chunks_path, os.path.basename(video_path).split(".")[0])+str(scene_list[chunk_number])+".mp4",
        fourcc, video_fps, (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    )

    while ret:
        if frame_number == scene_list[chunk_number]:
            chunk_to_write.release()
            chunk_number += 1
            chunk_to_write = cv2.VideoWriter(
                os.path.join(chunks_path, os.path.basename(video_path).split(".")[0])+str(scene_list[chunk_number])+".mp4",
                fourcc, video_fps, (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            )

        chunk_to_write.write(frame)
        frame_number += 1
        ret, frame = video.read()

    return [os.path.join(chunks_path, os.path.basename(video_path).split(".")[0])+str(chunk)+".mp4" for chunk in scene_list]


if __name__ == "__main__":
    start_time = time.time()
    path = 'videos/gopro_compilation.mp4'  # Path to video file.
    print(extract_chunks(path))
    print("Time for processing:")
    print(time.time()-start_time)
