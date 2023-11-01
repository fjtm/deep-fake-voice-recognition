import zipfile
import zlib
import os
import pandas as pd
import io
from typing import List

def save_data_zip(audio_files: List[str], i: int) -> None:
    """
    Save audio data to a ZIP archive after preprocessing and creating a CSV file.

    Args:
        audio_files (List[str]): List of audio file paths.
        i (int): An integer identifier used in file naming.

    Returns:
        None

    This function preprocesses the provided audio files, creates a CSV file, and then
    compresses it into a ZIP archive. The resulting ZIP file is saved to a specific location
    in Google Drive.

    Example:
    save_data_zip(['/path/to/audio1.wav', '/path/to/audio2.wav'], 1)
    """
    # Replace 'your_csv_file.csv' with the name of your CSV file
    csv_file_path = f'preprocess_data_{i}.csv'
    zip_file_path = f'/content/drive/My Drive/deep-fake-voice-recognition/data/preprocess_data_{i}.zip'

    # Preprocess audios
    preprocessed_audios = process_audio_files(audio_files, num_threads=None)

    # Save tmp data table
    pd.concat(preprocessed_audios, axis=0, ignore_index=True, sort=False).to_csv(csv_file_path)

    # Create a ZIP archive containing the CSV file with maximum compression
    with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.writestr(csv_file_path, zlib.compress(open(csv_file_path, 'rb').read()))

    # Clean up: Remove the temporary CSV file
    os.remove(csv_file_path)


def read_data_zip(zip_file_path: str, csv_encoding: str = 'utf-8') -> pd.DataFrame:
    """
    Read CSV data from a ZIP archive with deflated compression.

    Args:
        zip_file_path (str): Path to the ZIP file containing CSV data.
        csv_encoding (str): Encoding of the CSV file.

    Returns:
        pd.DataFrame: The DataFrame containing the data from the CSV file.

    Example:
    df = read_data_zip('/content/drive/My Drive/deep-fake-voice-recognition/data/preprocess_data_1.zip', csv_encoding='utf-8')
    """
    # Extract the CSV file from the ZIP archive
    with zipfile.ZipFile(zip_file_path, 'r') as zipf:
        # Assuming there is only one CSV file in the ZIP archive
        csv_file_name = zipf.namelist()[0]
        with zipf.open(csv_file_name) as csv_file:
            # Read the compressed data and then decompress it with zlib
            compressed_data = csv_file.read()
            decompressed_data = zlib.decompress(compressed_data).decode(encoding=csv_encoding)
            # Create a DataFrame from the decompressed data
            df = pd.read_csv(io.StringIO(decompressed_data))
    
    return df.iloc[:,1:]
