import os
import mne
import numpy as np
import scipy.io as io

from typing import Literal


def convert_gdf_to_mat(
        input_dir: str,
        output_dir: str,
        dataset_type: Literal['2a', '2b'] = '2a',
        data_type: Literal['T', 'E'] = 'T'
    ) -> None:
    """
    Converts GDF files from BCI Competition IV dataset to MAT format.

    :input_dir: Path to directory containing source GDF files.
    :output_dir: Path to directory where converted MAT files will be saved.
    :dataset_type: Type of BCI Competition dataset
    :data_type: Type of GDF files, "T" for train and "E" for eval dataset.
    """
    if dataset_type not in ['2a', '2b']:
        raise ValueError("Invalid dataset_type. Use '2a' or '2b'.")
    if data_type not in ['T', 'E']:
        raise ValueError("Invalid data_type. Use 'T' or 'E'.")

    eog_channels = ["EOG-left", "EOG-central", "EOG-right"]
    os.makedirs(output_dir, exist_ok=True)

    for subject_id in range(1, 10):
        if dataset_type == '2a':
            prefix = f"A{subject_id:02d}{data_type}"

            gdf_path = os.path.join(input_dir, f"BCICIV_{dataset_type}_gdf", f"{prefix}.gdf")
            label_path = os.path.join(input_dir, f"BCICIV_{dataset_type}_answers", f"{prefix}.mat")
            output_path = os.path.join(output_dir, f"{prefix}.mat")

            raw_data = mne.io.read_raw_gdf(gdf_path, preload=True)
            events, event_dict = mne.events_from_annotations(raw_data)

            event_ids = ({
                "Left": event_dict['769'],
                "Right": event_dict['770'], 
                "Foot": event_dict['771'],
                "Tongue": event_dict['772']
            } if data_type == "T" else {"Unknown": event_dict['783']})

            mask = np.array([event_id in list(event_ids.values()) for event_id in events[:, 2]])
            selected_events = events[mask]
            raw_data.info["bads"].extend(eog_channels)
            
            epochs = mne.Epochs(
                raw_data,
                selected_events,
                event_ids,
                picks=mne.pick_types(raw_data.info, eeg=True, exclude="bads"),
                tmin=0,
                tmax=3.996,
                preload=True,
                baseline=None
            )

            labels = (
                io.loadmat(label_path)["classlabel"] 
                if data_type == 'T' 
                else np.zeros(len(selected_events))
            )

            io.savemat(output_path, {
                "data": epochs.get_data(),
                "label": labels
            })
        else:
            sessions = range(1, 4) if data_type == 'T' else range(4, 6)
            for session in sessions:
                prefix = f"B{subject_id:02d}{session}{data_type}"
                gdf_path = os.path.join(input_dir, f"BCICIV_{dataset_type}_gdf", f"{prefix}.gdf")
                label_path = os.path.join(input_dir, f"BCICIV_{dataset_type}_answers", f"{prefix}.mat")
                output_path = os.path.join(output_dir, f"{prefix}.mat")

                raw_data = mne.io.read_raw_gdf(gdf_path, preload=True)
                events, event_dict = mne.events_from_annotations(raw_data)

                event_ids = {
                    "Left": event_dict['769'], 
                    "Right": event_dict['770']
                } if data_type == 'T' else {"Unknown": event_dict['783']}

                mask = np.array([event_id in list(event_ids.values()) for event_id in events[:, 2]])
                selected_events = events[mask]
                raw_data.info["bads"].extend(eog_channels)

                epochs = mne.Epochs(
                    raw_data,
                    selected_events,
                    event_ids,
                    picks=mne.pick_types(raw_data.info, eeg=True, exclude="bads"),
                    tmin=2,
                    tmax=6,
                    preload=True,
                    baseline=None
                )

                labels = (
                    io.loadmat(label_path)["classlabel"] 
                    if data_type == 'T' 
                    else np.zeros(len(selected_events))
                )

                io.savemat(output_path, {
                    "data": epochs.get_data(),
                    "label": labels
                })

if __name__ == "__main__":
    input_dir = "../raw_data"
    output_dir = "../data"

    data_types = ['T', 'E']
    dataset_types = ['2a', '2b']

    for data_type in data_types:
        for dataset_type in dataset_types:
            convert_gdf_to_mat(input_dir, output_dir, dataset_type, data_type)
