# X-intNMF

X-intNMF is the framework for multi-omics data analysis which combines integrative NMF with cross-interomics interaction network regularization for cancer subtype classification, prognosis prediction, and survival analysis.

## Installation

The required libraries are listed in the `requirements.txt` file. Users can install them using pip on usersr predefined Python virtual environment or conda environment:

```bash
pip install -r requirements.txt
```

## Sample dataset
Sample datasets are provided in the `sample_processed_data` folder. There are three subfolders:
- `BRCA_2omics` contains the breast cancer data with two omics data types: mRNA and miRNA, plus the mRNA-miRNA interaction network,
- `BRCA_3omics` contains the breast cancer data with three omics data types: mRNA, miRNA, and methylation, plus the mRNA-miRNA interaction network,
- `BRCA_micro_dataset` contains the reduced breast cancer data three omics data types: mRNA, miRNA, and methylation, plus the mRNA-miRNA interaction network, intended for quick testing.

All data files are in `*.parquet` format for read/write efficiency and DataFrame indexing. Additionally, two notebooks `sample_data_processing_BRCA.ipynb` and `sample_data_processing_LUAD.ipynb` are also provided to demonstrate how to process the raw data into the required format for X-intNMF.


## Data input & output format
- All the omics data must have the following format:
    - Each omics data is in **feature-by-sample** format, with sample names as columns and feature names as rows.
    - The columns of all omics data and its order must be the same.
- The interaction network is in **feature-by-feature** format, with feature names as both rows and columns. For example, if the interaction network is mRNA-miRNA interaction network, then the rows must be mRNA features and the columns must be miRNA features. The interaction network is binary (0-1) matrix, with 1 indicating the interaction between the feature pair. The interaction direction can be either one-way or two-way. If users only provide one-way interaction network, X-intNMF will automatically convert it to two-way interaction network by adding the transpose of the matrix to itself.
- The output of X-intNMF is a tuple of two elements:
    - The first element is a list of NumPy ndarrays, each NumPy ndarray contains the NMF factor matrix for each omics data type, which is in **feature-by-latent-factors** format, with features as rows and latent factors as columns.
    - The second element is a NumPy ndarray containing the NMF factor matrix for the sample data, which is in **latent-factors-by-sample** format, with latent factors as rows and samples as columns.
Although the output format are NumPy ndarrays, users can easily convert them to pandas DataFrame, which is demonstrated in the notebook.


## Additional information
- Core model is located in folder `model`, with most of functions, classes and methods are already commented and documented using docstrings.
- During the research, X-intNMF and its baselines were run along with the following support tools:
    - [MLFlow](https://mlflow.org/) for experiment tracking,
    - [MinIO](https://min.io/) for data, results, checkpoints storage
    - [MongoDB](https://www.mongodb.com/) for experiment result and parameters optimization storage.

For simplicity and easy access, the authors did not include these tools in the main model. However, some of its remaining code can be found in the baselines implementation. If users are interested in these integration, please contact the authors for more information.

- For GPU support, X-intNMF have options for selecting GPU and running flavors/backend. There are 3 available backend:
    - `numpy` for CPU only. All operations are performed using numpy.
    - `cupy` for GPU using cupy. Most of the iterative operations are performed using cupy, the rest are performed using numpy.
    - `pytorch` for GPU using pytorch. Most of the iterative operations are performed using pytorch, the rest are performed using numpy. **This is the recommended backend**


## Known issues
- Documentation for methods and classes on `__init__.py` file of `X-intNMF` class is not correctly rendered when hovering over the method/class name. Users can view the documentation by shift+click on the method/class name to view the source code and its docstring.

## Contact
For any concern or further assistance, please contact [tienthanh.bui@ucf.edu](mailto:tienthanh.bui@ucf.edu)


(c) 2025 bu1th4nh / UCF Computational Biology Lab. All rights reserved. 