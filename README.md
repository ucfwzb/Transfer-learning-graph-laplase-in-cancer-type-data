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
Download four .pk files(sample_breast_expression.pk, sample_breast_label.pk, sample_ov_expression.pk, sample_ov_label.pk) and two .py files(model1.py, model2.py) into the same fold. Turn on the terminator and move the path to the fold, run python command and execute:
`python model1.py` or `python model2.py`
The results will be exported as two txt files in the same fold. 

## File description 
* **sample_ov_expression.pk:** This is a sample file of **Ovarian Cancer** gene expression data set. The size of the data is 50 x 200, the 50 is the number of samples, the 500 is the number of genes. The value range is [0,20.4273].

* **sample_ov_label.pk:** This is a sample file of **Ovarian Cancer** label data. The size of it is 50.

* **sample_breast_expression.pk:** This is a sample file of **Breast Cancer** gene expression set. The size of the data is 50 x 200, the 50 is the number of samples, the 500 is the number of genes. The value range is [0,20.3229].

* **sample_breast_label.pk:** This is a sample file of **Breast Cancer** label data. The size of it is 50.

* **model[1,2].py:** This is the execution file, including the model[1,2]'s realization codes.

## Data clean 
In order to reduce data noise, the genes whose values' mean and standard deviation values are at low level in both datasets should be removed. The initial level threshold is 50%, and it can be changed in the read_data function of the models.py file.

## Cross-validation
All models will run though the same cross validation process together. For both data sets, 20 samples are randomly selected for test part, 20 samples for validation part, the rest samples are training part. This process will run for 50 times. and the average performance on test data set is set as model's performance. 
**Warning:** Inappropriate parameter selection will cause model generating bad result, and in this situation, model will set the ROC result as 0 or 0.5. Please remove this result and corresponding parameters.









