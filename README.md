### Code for On the Impact of Algorithmic Recourse on Social Segregation

For setting up general environment and IBR environement, run

```
conda create --name recourse python=3.6.13 pip
conda activate recourse 
pip install -r requirements_g.txt
```

Since EBR use the cde package which has conflict with the general environment, you need to create a new environment and run 

```
conda create --name cde python=3.6.13 pip
conda activate cde 
pip install -r requirements_ebr.txt
```

The toy example is included in the [jupyter notebook](/balanced_recourse.ipynb). 

The implementation of IBR is included in balancevae.py. You need to create a method in the carla package under ./carla/recourse_methods/autoencoder/models/ to run the method. An example of EBR is included in EBR.ipynb. 


To run experiments on GMC-Age and Law-Sex, run 

```
python3 main_impacted_data.py --ar_model wachter --ml_model ann --dataset give_me_some_credit --sens age 
python3 main_impacted_data.py --ar_model CCHVAE --ml_model ann --dataset give_me_some_credit --sens age 

# IBR
python3 main_impacted_data.py --ar_model balanceCCHVAE --ml_model ann --dataset give_me_some_credit --sens age --gamma 200

# EBR
python3 main_impacted_data_clean.py --ar_model balanced --ml_model ann --dataset give_me_some_credit --sens age --balancedata 1 --balancedatastage 0
# in (cde) environment 
python3 use_balance.py --dataset give_me_some_credit --sens age --nepochs 1000
# back in (recourse) environment 
python3 main_impacted_data_clean.py --ar_model balanced --ml_model ann --dataset give_me_some_credit --bq 0.7 --balancedata 1 --balancedatastage 1 --sens age
```

Results:
Methods |Metric | GMC (Origin) | GMC (After) | Law (Origin) |Law (After) |
| --- | --- | --- | ----------- | ----------- | ----------- |
|Wachter | Centrality |     0.341| <font color='green'>0.342</font> |0.086|0.083|
|Wachter | Atkinson Index | 0.122| 0.122 |0.032|<font color='green'>0.033</font>|
|Wachter | Avg Prox |       0.582| 0.582 |0.703|0.707|
|CCHVAE | Centrality |      0.341| 0.333 |0.086|0.074|
|CCHVAE | Atkinson Index |  0.122| <font color='green'>0.125</font> |0.032|<font color='green'>0.034</font>|
|CCHVAE | Avg Prox |        0.582| 0.588 |0.703|0.717|
|IBR| Centrality |          0.341| 0.332 |0.086|0.069
|IBR| Atkinson Index |      0.122| 0.119 |0.032|0.032
|IBR| Avg Prox |            0.582| 0.590 |0.703|0.718
|EBR| Centrality |          0.341| 0.336 |0.086|0.083
|EBR| Atkinson Index |      0.122| 0.122 |0.032|0.032
|EBR| Avg Prox |            0.582| 0.585 |0.703|0.706

