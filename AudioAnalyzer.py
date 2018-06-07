import libs.librosa as librosa
import numpy as np


class AudioAnalyzer:

    def __init__(self, audio_path):
        self.audio_path = audio_path
        self.pks_timestamps = []
        self.pks_frames = []
        self.chunk_features = []
        self.features = None
        self.processed = False

    @staticmethod
    def extract_features(y, sr):
        features = np.zeros(48, dtype=np.float32)
        S = np.abs(librosa.stft(y))
        tempo, beats = librosa.beat.beat_track(y, sr)
        features[0] = tempo
        features[1] = sum(beats)
        features[2] = np.average(beats)
        # std, mean and var as features
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_cq = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
        melspectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
        rmse = librosa.feature.rmse(y=y)
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        poly_features = librosa.feature.poly_features(S=S, sr=sr)
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        harmonic = librosa.effects.harmonic(y)
        percussive = librosa.effects.percussive(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)

        values_to_process = [chroma_stft, chroma_cq, chroma_cens, melspectrogram, rmse,
                             cent, spec_bw, contrast, rolloff, poly_features, tonnetz,
                             zcr, harmonic, percussive, mfcc]

        for i, value in enumerate(values_to_process):
            features[3+i*3] = np.mean(value)
            features[4+i*3] = np.std(value)
            features[5+i*3] = np.var(value)

        return features

    @staticmethod
    def peaks_timestamps(y, sr, pre_max=10, post_max=10, pre_avg=15, post_avg=15, delta=2.0, wait=40):
        onset_env = librosa.onset.onset_strength(y=y, sr=sr,
                                                 hop_length=512,
                                                 aggregate=np.median)
        times = librosa.frames_to_time(np.arange(len(onset_env)))
        peaks = librosa.util.peak_pick(onset_env, pre_max, post_max, pre_avg, post_avg, delta, wait)
        peak_timestamps = [times[peak] for peak in peaks]
        return peak_timestamps

    # def save_timestamps(self):
    #     to_save_path = os.path.join(self.out_path, os.path.basename(self.audio_path).split(".")[0]+".txt")
    #     with open(to_save_path, "w") as f:
    #         for t in self.pks_timestamps:
    #             f.write(str(t)+"\n")

    def get_general_features(self):
        if not self.processed:
            self.run()
        return self.features

    def get_chunks_features(self):
        if not self.processed:
            self.run()
        return self.chunk_features

    def get_timestamps(self):
        """:return timestamps in msecs"""
        if not self.processed:
            self.run()
        return [1000*i for i in self.pks_timestamps]

    def get_min_max_intervals(self):
        """
        Max and min chunks in msecs
        """
        if not self.processed:
            self.run()
        lst = 0
        differences = []
        for tm in self.pks_timestamps:
            differences.append(tm-lst)
            lst = tm
        return 1000*min(differences), 1000*max(differences)

    def run(self):
        y, sr = librosa.load(self.audio_path)
        self.features = self.extract_features(y, sr)
        self.pks_timestamps = self.peaks_timestamps(y, sr)
        self.pks_frames = librosa.time_to_frames(times=self.pks_timestamps, sr=sr)
        last_frm = 0
        for i, frm in enumerate(self.pks_frames):
            if frm-last_frm < 10:
                self.pks_timestamps.pop(i-len(self.pks_frames)+len(self.pks_timestamps))
            else:
                self.chunk_features.append(self.extract_features(y[last_frm:frm], sr))
                last_frm = frm
        self.pks_frames = librosa.time_to_frames(times=self.pks_timestamps, sr=sr)
        self.processed = True
        return self.pks_timestamps


if __name__ == "__main__":
    test_path = "music/superorganism something-for-your-m-i-n-d.wav"
    a = AudioAnalyzer(test_path)
    a.run()