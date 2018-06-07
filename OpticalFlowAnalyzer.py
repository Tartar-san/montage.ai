import cv2
import numpy as np


class OpticalFlowAnalyzer:

    def getOpticalFlowMagnitude(self, path_to_video):
        cap = cv2.VideoCapture(path_to_video)
        ret, frame1 = cap.read()
        frame1 = cv2.resize(frame1, (240, 160))
        prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)

        mean_mags = []
        while ret:
            ret, frame2 = cap.read()

            if not ret:
                break
            frame2 = cv2.resize(frame2, (240, 160))
            next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            mean_mags.append(np.mean(mag))
            prvs = next
        cap.release()
        return np.max(mean_mags)


if __name__ == '__main__':
    optical_flow_analyzer = OpticalFlowAnalyzer()
    print("Mean optical flow magnitude: ", optical_flow_analyzer.getOpticalFlowMagnitude("/Users/vbudzan/Downloads/videoplayback.mp4"))
    print("Mean optical flow magnitude: ", optical_flow_analyzer.getOpticalFlowMagnitude("/Users/vbudzan/Downloads/Dubstep Bird (Original 5 Sec Video).mp4"))
    print("Mean optical flow magnitude: ", optical_flow_analyzer.getOpticalFlowMagnitude("/Users/vbudzan/Downloads/Nature Beautiful short video 720p HD.mp4"))