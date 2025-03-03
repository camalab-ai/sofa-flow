# sofa-flow: Streamed optical flow adaptation from synthetic to real dental scenes


## Usage

We provide scripts and code to evaluate and generate datasets for Streamed optical flow adaptation. 

**Metrics implementation**: 
```shell
python3 infer_TMI_metrics.py
```

**Implementation of main class for metric computation and metric based optical flow selection**: 
```shell
python3 infer_TMI_seq.py
```

**Optical Flow Inference and data preparation for Streamed optical flow training**: 

This script performs optical flow estimation for image sequences using the RAFT or GMA model and saves the processed data.

```shell
# To run the script, use the following command:
python infer_flow_TMI.py --DEVICE cuda --MODEL checkpoints/RAFT_Sintel.pth --DATA_PATH /path/to/dataset --SPLIT test --IMAGE_DIR GT --SAVE_DATASET_DIR TMI_results

# Example usage
python infer_flow_TMI.py --DEVICE cuda --DATA_PATH /DATASETS/Vident-real-100 --SPLIT test --IMAGE_DIR GT --SAVE_DATASET_DIR output_results
```

How It Works
1. The script loads the model and input images (PNG files).
2. It computes the optical flow between consecutive frames.
3. The results are saved in the specified directory.
4. The script generates and saves metric plots based on the results.
Output files are stored in the directory specified by --SAVE_DATASET_DIR.

**_________________________________________________________________________**




