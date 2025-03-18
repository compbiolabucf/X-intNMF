# X-intNMF

X-intNMF is the framework for multi-omics data analysis which combines integrative NMF with cross-interomics interaction network regularization for cancer subtype classification, prognosis prediction, and survival analysis.

## Installation

The required libraries are listed in the `requirements.txt` file. You can install them using pip on your predefined Python virtual environment or conda environment:

```bash
pip install -r requirements.txt
```

## Sample dataset

Sample datasets are provided in the `sample_processed_data` folder. There are three subfolders:
- `BRCA_2omics` contains the breast cancer data with two omics data types: mRNA and miRNA, plus the mRNA-miRNA interaction network,
- `BRCA_3omics` contains the breast cancer data with three omics data types: mRNA, miRNA, and methylation, plus the mRNA-miRNA interaction network,
- `BRCA_micro_dataset` contains the reduced breast cancer data three omics data types: mRNA, miRNA, and methylation, plus the mRNA-miRNA interaction network, intended for quick testing.

Additionally, two notebooks `sample_data_processing_BRCA.ipynb` and `sample_data_processing_LUAD.ipynb` are also provided to demonstrate how to process the raw data into the required format for X-intNMF.

## Code usage
- File `X-intNMF-main.ipynb` contains the example code for running X-intNMF on the sample datasets.


## Baselines and downstream tasks
- Folder `baselines` contains the modified code for baseline methods, including MOFA2, MOGONET, MCRGCN, and MOFA2.
- Folder `downstream` contains the code for downstream tasks, including survival analysis and prognosis prediction. Detailed instructions are provided upon request.


## Additional information
- Core model is located in folder `model`, with most of functions, classes and methods are already documented using docstrings.
- During the research, X-intNMF and its baselines were run along with the following support tools:
    - [MLFlow](https://mlflow.org/) for experiment tracking,
    - [MinIO](https://min.io/) for data, results, checkpoints storage
    - [MongoDB](https://www.mongodb.com/) for experiment result and parameters optimization storage.

For simplicity and easy access, the authors did not include these tools in the main model. However, some of its remaining code can be found in the baselines implementation. If you are interested in these integration, please contact the authors for more information.

- For GPU support, X-intNMF have options for selecting GPU and running flavors/backend. There are 3 available backend:
    - `numpy` for CPU only. All operations are performed using numpy.
    - `cupy` for GPU using cupy. Most of the iterative operations are performed using cupy, the rest are performed using numpy.
    - `pytorch` for GPU using pytorch. Most of the iterative operations are performed using pytorch, the rest are performed using numpy. **This is the recommended backend**


(c) 2025 bu1th4nh / UCF Computational Biology Lab. All rights reserved. 