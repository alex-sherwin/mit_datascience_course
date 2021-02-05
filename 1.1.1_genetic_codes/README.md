# Python Libraries

```bash
pip3 install pandas psycopg2-binary scikit-learn Matplotlib scikit-image seaborn Cython
```


# Data Set

The original data set is `genetic_codes_200_per_line`, obtained from http://www.ihes.fr/~zinovyev/pcadg/ccrescentus.fa

The original data is encoded with 200 characters per line, but the MITxPRO course says to use 300, so the data has been converted 
from 200 characters per line to a single line, and then broken into 300 characters per line, to get the expected 1,018 lines of DNA
sequence in `genetic_codes_300_per_line`


# Running

```bash
python3 genetic_codes.py
```