
import numpy as np
import librosa
from typing import Optional, Any, List

class AudioProperties:
    def __init__(self, audio_path: str, sr: Optional[int] = None, **kwargs: Any):
        """
        Initialize the AudioProperties.

        Args:
            audio_path (str): Path to the wav audio to extract properties.
            sr (int): Sampling rate (None by default).
            **kwargs: Additional keyword arguments to pass to librosa.feature.
        """
        self.audio_path = audio_path
        self.kwargs = kwargs
        self.y, self.sr = librosa.load(self.audio_path, sr=sr)

    def get_chromogram(self, n_chroma: int = 12, **kwargs: Any) -> np.ndarray:
        """
        Create a np.ndarray with dimension n_chroma x sr.

        Args:
            n_chroma (int): Number of chroma bins to produce (12 by default).
            **kwargs: Additional keyword arguments to pass to librosa.feature.

        Returns:
            np.ndarray: Chromagram representation of the audio.
        """
        chroma = librosa.feature.chroma_stft(y=self.y, sr=self.sr, n_chroma=n_chroma, **kwargs)
        return chroma

    def get_rms(self, **kwargs: Any) -> np.ndarray:
        """
        Calculate the Root Mean Square (RMS) of the audio.

        Args:
            **kwargs: Additional keyword arguments to pass to librosa.feature.rms.

        Returns:
            np.ndarray: RMS values.
        """
        rms = librosa.feature.rms(y=self.y, **kwargs)
        return rms

    def get_mfccs(self, n_mfcc: int = 40, dct_type: int = 1, **kwargs: Any) -> np.ndarray:
        """
        Calculate the Mel-Frequency Cepstral Coefficients (MFCCs) of the audio.

        Args:
            n_mfcc (int): Number of MFCCs to compute (40 by default).
            dct_type (int): Type of Discrete Cosine Transform to use (1 by default).
            **kwargs: Additional keyword arguments to pass to librosa.feature.mfcc.

        Returns:
            np.ndarray: MFCC coefficients.
        """
        mfccs = librosa.feature.mfcc(y=self.y, sr=self.sr, n_mfcc=n_mfcc, dct_type=dct_type, **kwargs)
        return mfccs

    def get_spectral_centroid(self, **kwargs: Any) -> np.ndarray:
        """
        Calculate the Spectral Centroid of the audio.

        Args:
            **kwargs: Additional keyword arguments to pass to librosa.feature.spectral_centroid.

        Returns:
            np.ndarray: Spectral Centroid values.
        """
        cent = librosa.feature.spectral_centroid(y=self.y, sr=self.sr, **kwargs)
        return cent

    def get_spectral_bandwidth(self, **kwargs: Any) -> np.ndarray:
        """
        Calculate the Spectral Bandwidth of the audio.

        Args:
            **kwargs: Additional keyword arguments to pass to librosa.feature.spectral_bandwidth.

        Returns:
            np.ndarray: Spectral Bandwidth values.
        """
        spec_bw = librosa.feature.spectral_bandwidth(y=self.y, sr=self.sr, **kwargs)
        return spec_bw

    def get_spectral_rolloff(self, roll_percent: List[float] = [0.01, 0.5, 0.99], **kwargs: Any) -> np.ndarray:

        """
        Calculate the Spectral Roll-off of the audio for multiple roll_percent values.

        Args:
            roll_percent (List[float]): List of roll-off percentages ([0.01, 0.5, 0.99] by default).
            **kwargs: Additional keyword arguments to pass to librosa.feature.spectral_rolloff.

        Returns:
            np.ndarray: Spectral Roll-off values for each roll-off percentage.
        """
        rolloff_values = []
        for percent in roll_percent:
            rolloff = librosa.feature.spectral_rolloff(y=self.y, sr=self.sr, roll_percent=percent, **kwargs)
            rolloff_values.append(rolloff)

        return np.array(rolloff_values)
