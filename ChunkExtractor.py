import scenedetect
import time
import cv2
import os
import ffmpy

from config import *


class ChunkExtractor:

    def __init__(self, min_len=500, max_len=6000):
        """
        Min and max length are in msecs
        """
        self.min_len = min_len
        self.max_len = max_len
        self.detectors = [
            scenedetect.detectors.ThresholdDetector(threshold=16, min_percent=0.9, min_scene_len=min_len/40),
            scenedetect.detectors.ContentDetector(min_scene_len=min_len/40)
        ]

    @staticmethod
    def convert_msecs_to_ffmpeg_time(msecs):
        """:return str in format hh:mm:ss"""
        time = ""
        time += str(int(msecs / 1000 / 60 / 60))+":"
        time += str(int(msecs / 1000 / 60) % 60) + ":"
        time += str((int(msecs) % 60000) / 1000)
        return time

    def extract_chunks(self, video_path, chunks_path=CHUNKS_PATH):
        scene_list = []

        video_fps, frames_read = scenedetect.detect_scenes_file(
            video_path, scene_list, self.detectors)

        scene_list.sort()
        scene_list_msec = [(1000.0 * x) / float(video_fps) for x in scene_list]

        chunks_list = []

        last = 0
        chunk_number = 0
        for time in scene_list_msec:
            if time-last < self.min_len:
                continue
            elif time-last > self.max_len:
                subchunks_number = int((time-last) / self.max_len) + 1
                step = (time-last)/subchunks_number
                now_time = last+step
                while now_time <= time:
                    out_filename = os.path.basename(video_path).split(".")[0] + str(chunk_number) + "." + \
                                   os.path.basename(video_path).split(".")[1]
                    out_path = os.path.join(chunks_path, out_filename)
                    ffmpeg_start_time = self.convert_msecs_to_ffmpeg_time(last)
                    ffmpeg_end_time = self.convert_msecs_to_ffmpeg_time(now_time)
                    ff = ffmpy.FFmpeg(
                        inputs={video_path: None},
                        outputs={out_path:["-ss", ffmpeg_start_time, "-to", ffmpeg_end_time, "-y"]}
                    )
                    ff.run()
                    chunks_list.append([out_path, now_time-last])
                    last = now_time
                    now_time += step
                    chunk_number += 1
            else:
                out_filename = os.path.basename(video_path).split(".")[0] + str(chunk_number) + "." + \
                               os.path.basename(video_path).split(".")[1]
                out_path = os.path.join(chunks_path, out_filename)
                ffmpeg_start_time = self.convert_msecs_to_ffmpeg_time(last)
                ffmpeg_end_time = self.convert_msecs_to_ffmpeg_time(time)
                ff = ffmpy.FFmpeg(
                    inputs={video_path: None},
                    outputs={out_path: ["-ss", ffmpeg_start_time, "-to", ffmpeg_end_time, "-y"]}
                )
                ff.run()
                chunks_list.append([out_path, time-last])
                last = time
                chunk_number += 1

        # video = cv2.VideoCapture(video_path)
        # ret, frame = video.read()
        # frame_number = 0
        # chunk_number = 0
        #
        # fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # chunk_to_write = cv2.VideoWriter(
        #     os.path.join(chunks_path, os.path.basename(video_path).split(".")[0])+str(scene_list[chunk_number])+".mp4",
        #     fourcc, video_fps, (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        # )
        #
        # while ret:
        #     if frame_number == scene_list[chunk_number]:
        #         chunk_to_write.release()
        #         chunk_number += 1
        #         chunk_to_write = cv2.VideoWriter(
        #             os.path.join(chunks_path, os.path.basename(video_path).split(".")[0])+str(scene_list[chunk_number])+".mp4",
        #             fourcc, video_fps, (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        #         )
        #
        #     chunk_to_write.write(frame)
        #     frame_number += 1
        #     ret, frame = video.read()

        # return [os.path.join(chunks_path, os.path.basename(video_path).split(".")[0])+str(chunk)+".mp4" for chunk in scene_list]
        return chunks_list

if __name__ == "__main__":
    chunker = ChunkExtractor()
    start_time = time.time()
    path = 'videos/Gorillaz - Humility (Official Video).mp4'  # Path to video file.
    print(chunker.extract_chunks(path))
    print("Time for processing:")
    print(time.time()-start_time)
