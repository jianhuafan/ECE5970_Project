# ECE5970 Class Project & TADPOLE Challenge



## Models
SVM, 
LSTM


## Results on train, test, validation data (LSTM)

| data set | Accuracy of Diagnosis |MAE for ADAS|MAE for Ventricles Norm|MAE for MMSE|
|--------|:---------:|:------:|:------:|:------:|
| train | 93.3% | 2.21 | 0.0078 | 3.58 |
|--------|:---------:|:------:|:------:|:------:|
| test | 77.5% | 4.93 | 0.0096 | 3.99 |
|--------|:---------:|:------:|:------:|:------:|
| train | 77.0% | 4.95 | 0.0084 | 4.23 |


## Usage

### Train of LSTM

1. library requirement: Tensorflow, matplotlib

2. run 
```
$python train.py
```

3. additional parameters can be added like 
```
$python train.py  —max_epochs 100
```
type
```
$python train.py —help
```
to see more details.
4. train_reg.py is used for prediction of continuous variables. Which variable should be specified when run it, otherwise it will predict MMSE by default. type
```
$python train_reg.py —-help for more detail.
```

### Test and validation of LSTM

1. run
```
$python test.py
```
for categorical data. To test continuous variables, run
```
$python test_reg.py
```
Similarly, parameters can be specified by adding additional arguments. see help command for more details.
2. To validate, replace the input test path and output test path with corresponding validation data path.

###


