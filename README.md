# GIL - Generalized Image Learning 

## Purpose
This repo houses the initial scripts for building a deep learning app on BioData Catalyst powered by Seven Bridges. These scripts will be used to test training scalability, among other issues.

## Main script
`train.py` creates a VGG-16 model for single-channel image classification.
### Input arguments:
| Arg | Description | Type | Values | Required |
| --- | ----------- | ---- | ------ | -------- |
| --data_csv | Path to CSV file pointing to images/labels | string |  | YES |
| --image_column | Column name for images | string |  | YES |
| --label_column | Column name for labels | string |  | YES |
| --test_ratio | Percentage for testing data | float | 0.3 (Default) |   |
| --epochs | Number of training epochs | int | 15 (Default) |   |
| --batch_size | Training batch size | int | 8 (Default) |   |
| --output | Specify file name for output | string | 'model' (Default) |   |

## Debug
`get_sizes.py --data_csv /path/to/file.csv --image_column image_path_column_name` will create a CSV containing the image name, SimpleITK image shape, and Numpy array shape. It will also print this information to the console.
