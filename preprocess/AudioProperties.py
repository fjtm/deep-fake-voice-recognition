
import numpy as np
import pandas as pd
import librosa
from typing import Optional, Any, List, Dict
import concurrent.futures
import os
from IPython.display import clear_output
import time

class AudioProperties():
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
            **kwargs: Additional keyword arguments to pass to librosa.feature.chroma_stft

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
        rolloff_values = np.array([
            librosa.feature.spectral_rolloff(y=self.y, sr=self.sr, roll_percent=percent, **kwargs).reshape(-1) 
            for percent in roll_percent
            ])

        return rolloff_values


class ProcessAudio(AudioProperties):
    def __init__(
        self,
        audio_path: str,
        sr: Optional[int] = None,
        methods: Optional[List[str]] = [
            "chromogram",
            "rms",
            "mfccs",
            "spectral_centroid",
            "spectral_bandwidth",
            "spectral_rolloff"
        ],
        **kwargs: Any
    ):
        """
        Initialize the ProcessAudio class.

        Args:
            audio_path (str): Path to the audio file for property extraction.
            sr (Optional[int]): Sampling rate (default: None, will use librosa default).
            n_mfcc (int): Number of MFCCs to compute (default: 40).
            methods (Optional[List[str]]): List of analysis methods to apply (default: ["chromogram", "rms", ...]).
            **kwargs: Additional keyword arguments to pass to the superclass constructor.
        """
        super().__init__(audio_path, sr, **kwargs)
        self.methods = methods

        if self.methods:
            self.check_available_methods()

    def check_available_methods(self):
        """
        Check if the specified methods are available for analysis.
        
        Raises:
            ValueError: If any of the specified methods is not available.
        """
        self.list_available_methods = [
            "chromogram",
            "rms",
            "mfccs",
            "spectral_centroid",
            "spectral_bandwidth",
            "spectral_rolloff"
        ]

        list_not_available_methods = [
            method
            for method in self.methods
            if method not in self.list_available_methods
        ]

        if list_not_available_methods:
            raise ValueError(f"Methods: {list_not_available_methods} not available. Available methods: {self.list_available_methods}")

    def transform(
        self,
        n_mfcc: int = 40,
        n_chroma: int = 12,
        roll_percent: List[float] = [0.01, 0.5, 0.99],
        **kwargs: Any
    ) -> Dict[str, np.ndarray]:
        """
        Transform audio data using specified analysis methods.

        Args:
            n_mfcc (int): Number of MFCCs to compute (default: 40).
            n_chroma (int): Number of chroma bins to produce (default: 12).
            roll_percent (List[float]): List of roll-off percentages (default: [0.01, 0.5, 0.99]).
            **kwargs: Additional keyword arguments to pass to analysis methods.

        Returns:
            Dict[str, np.ndarray]: A dictionary containing the results of the specified analysis methods.
        """
        if not self.methods:
            self.methods = self.list_available_methods

        data_transformed = {}

        if "chromogram" in self.methods:
            data_transformed["chromogram"] = super().get_chromogram(n_chroma, **kwargs)

        if "rms" in self.methods:
            data_transformed["rms"] = super().get_rms(**kwargs)

        if "mfccs" in self.methods:
            data_transformed["mfccs"] = super().get_mfccs(n_mfcc, **kwargs)

        if "spectral_centroid" in self.methods:
            data_transformed["spectral_centroid"] = super().get_spectral_centroid(**kwargs)

        if "spectral_bandwidth" in self.methods:
            data_transformed["spectral_bandwidth"] = super().get_spectral_bandwidth(**kwargs)

        if "spectral_rolloff" in self.methods:
            data_transformed["spectral_rolloff"] = super().get_spectral_rolloff(roll_percent, **kwargs)

        return self.create_dataframe_from_dict(data_transformed)

    def get_label(self) -> str:

        """
        Get string label from path.

        Args:
            None
        Returns:
            self.label (str): REAL or FAKE label
        """
        self.label = self.audio_path.split("/")[2]
        return self.label

    def get_index(self) -> str:
        """
        Create a pandas DataFrame from a dictionary of NumPy arrays with specific column naming.

        Args:
            None
        Returns:
            self.ind (str): Name index of the audio
        """ 
        self.ind = self.audio_path.split("/")[-1].replace(".wav","")
        return self.ind

    def create_dataframe_from_dict(self, data_dict: Dict[str, np.ndarray]) -> pd.DataFrame:
        """
        Create a pandas DataFrame from a dictionary of NumPy arrays with specific column naming.

        Args:
            data_dict (Dict[str, np.ndarray]): A dictionary containing string keys and NumPy arrays as values.

        Returns:
            pd.DataFrame: A DataFrame with columns named as specified in the dictionary.
        """
        # Initialize a dictionary to store column data
        column_data = {}

        # Iterate through the keys and values in the dictionary
        for key, value in data_dict.items():
            # Get the number of columns in the array and create column names
            num_columns = value.shape[0]
            column_names = [f"{key}_{i+1}" for i in range(num_columns)]

            # Store column data in the dictionary
            for i, col_name in enumerate(column_names):
                column_data[col_name] = value[i, :]

        # Create a pandas DataFrame from the dictionary
        df_audio = pd.DataFrame(column_data)

        # Add data label
        self.get_label() 
        df_audio["label"] = [self.label]*df_audio.shape[0]

        # Add data index
        self.get_index()
        df_audio["ind"] = [self.ind]*df_audio.shape[0]

        return df_audio

def process_audio_files(audio_files: List[str], num_threads: int = None) -> List[pd.DataFrame]:
    """
    Process a list of audio files in parallel using a ThreadPoolExecutor.

    Args:
        audio_files (list): List of audio file paths to be processed.
        num_threads (int, optional): Number of threads to use. If not specified, it will use the CPU core count.

    Returns:
        list: List of results from processing each audio file.

    Raises:
        Exception: Any exception raised during audio processing.

    Example:
        audio_files = ["file1.wav", "file2.wav"]
        results = process_audio_files(audio_files)
    """
    # Get the number of CPU cores
    num_cores = num_threads if num_threads is not None else os.cpu_count()

    # Create a ThreadPoolExecutor with the specified or default number of threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_cores) as executor:
        # Submit audio processing tasks and keep track of futures
        futures = [executor.submit(lambda audio_file: ProcessAudio(audio_file).transform(), audio_file) for audio_file in audio_files]

        # Track progress
        completed = 0
        total = len(futures)

        # Start the timer
        start_time = time.time()

        results = []

        for future in concurrent.futures.as_completed(futures):
            completed += 1
            progress = (completed / total) * 100

            # Clear the output before each print
            clear_output(wait=True)

            print(f"Progress: {completed}/{total} ({progress:.2f}%)")

            try:
                # Get the result of the completed task and store it in the results list
                result = future.result()
                results.append(result)
            except Exception as e:
                # Handle any exception raised during audio processing
                results.append(e)
                print(f"Error processing audio: {e}")

        # Stop the timer
        end_time = time.time()

    print("Processing completed.")
    print(f"Total time taken: {end_time - start_time:.2f} seconds")

    return results
