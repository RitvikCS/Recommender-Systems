**********RECOMMENDER SYSTYEMS************

**********STEPS FOR USING THE PROGRAM**********
1) Run preprocess.py and wait until it asks to enter a query.
2) Run respective files for collaborative, collaborative with baseline, svd, svd with 90% energy retain, cur and cur with 90%energy    retain.
3) RMSE, Spearman Correlation, precision at rank K and time required for prediction will be shown on successfull completion.

*************************

**********LOAD CUSTOM DATASETS INTO PROGRAMS**********
=> copy the dataset files into the folder containing the python files.
=> open preprocess.py source file,
=> in driver code...
=> change the value of path to your custom dataset file in Dataset folder. "Dataset must be in .dat format".


=>Preprocessing the data takes sometime as generally the datasets are huge in number.
=>You will need NLTK package, Numpy package for successful compilation.
=>Following python modules are required :- pandas, numpy, pickle, sklearn.
