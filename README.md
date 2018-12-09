# Transfer-learning-graph-laplase-in-cancer-type-data
code for bioinformatics submission

## Softerware requirement
- Python 3.6 and up
- packages:
- - Numpy
- - Pandas
- - Pickle
- - scikit-learn

## How to run the project
Download ov_total.zip, brca_total.zip, models.py into the same fold and decompress two zip files in the fold too. Turn on the termenator and move the path to the fold, run python command and execute:
`python models.py`

## File description 
* **ov_total.pkl(ov_total.zip):** This is **Ovarial Cancer** gene expression data set. The size of the data is 252 x 20531, the 252 is the number of samples, the 20531 is the number of genes and the label, and the first column is the label data. The value range is [0,20.4273].

* **brca_total.pkl(brca_total.zip):** This is **Breast Cancer** gene expression set. The size of the data is 198 x 20531, the 198 is the number of samples, the 20531 is the number of genes and the label, and the first column is the label data. The value range is [0,20.3229].

* **models.py:** This is the execution file, including all the models' realization codes.

## Data clean 
In order to reduce data noise, the genes whose values' mean and standard deviation values are at low level in both datasets should be removed. Initial level threshold is 50%. 

## Cross-validation








