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
After long time running, the result will be exported as a txt file in the same fold.

## File description 
* **ov_total.pkl(ov_total.zip):** This is **Ovarial Cancer** gene expression data set. The size of the data is 252 x 20531, the 252 is the number of samples, the 20531 is the number of genes and the label, and the first column is the label data. The value range is [0,20.4273].

* **brca_total.pkl(brca_total.zip):** This is **Breast Cancer** gene expression set. The size of the data is 198 x 20531, the 198 is the number of samples, the 20531 is the number of genes and the label, and the first column is the label data. The value range is [0,20.3229].

* **models.py:** This is the execution file, including all the models' realization codes.

## Data clean 
In order to reduce data noise, the genes whose values' mean and standard deviation values are at low level in both datasets should be removed. The initial level threshold is 50%, it can be changed in the read_data function of the models.py file.

## Cross-validation
All models will run though the same cross validation process together. For both data sets, 20 samples are randomly selected for test part, 20 samples for validation part, the rest samples are training part. This process will run for 50 times. and the average performance on test data set is set as model's performance. 









