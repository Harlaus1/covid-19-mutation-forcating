# covid-19-mutation-forcating
The code is consist of 2 parts. First one is the calculation of mutation rate of covid-19 sequences, which can be downloaded from gsaid. Second one is the forcasting of future mutation rate. The forcasting model is a simple lstm.
In order to use the code, there are a few steps you need to follow.
First, download fasta files(covid-19 sequences) from gisaid, where you can filter sequences.
Second, use the first code "EachMutationRate.py". This code will count the overall substitutional mutation. The code will save 2 csv files recording mutations mentioned above. However, in the purpose of simplification, this programme ignored the insertion and deletion mutatuin considering of the complexity.
Third, run the "forcasting.py". It will save history training and validation losses. Also, 100 steps of future mutation rate can be found in the "prediction.csv"




The version of the pycharm and python are 2020.2.3 and 3.7 
