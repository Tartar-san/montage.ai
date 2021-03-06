from PIL import Image
import numpy as np
import os
import cv2
import tensorflow as tf

from libs.youtube.feature_extractor.feature_extractor import YouTube8MFeatureExtractor
os.environ["CUDA_VISIBLE_DEVICES"]="0"


class VideoFeatureExtractor:

    def __init__(self):
        self.extractor = YouTube8MFeatureExtractor()

    def extract_1024_features_from_frame(self, frame):
        features = self.extractor.extract_rgb_frame_features(frame)
        return features

    @staticmethod
    def compute_features_statistics(features):
        result = features.mean(axis = 0)
        result = np.concatenate((result, features.std(axis = 0)), axis = 0)
        return result

    def extract_features_from_video(self, path_to_video):
        cap = cv2.VideoCapture(path_to_video)
        features = []
        i = 0
        while cap.isOpened():
            i = i + 1
            ret, frame = cap.read()

            if not ret:
                break

            if i % 10 == 0:
                features.append(self.extract_1024_features_from_frame(frame))

        features = self.compute_features_statistics(np.asarray(features))

        return features

if __name__ == '__main__':

    feature_extractor = VideoFeatureExtractor()

    video_features = feature_extractor.extract_features_from_video("scrapped_videos/薛之謙 Joker Xue【肆無忌憚】HD 高清官方歌詞版 MV.mp4")

    print(video_features.shape)
    print(video_features)