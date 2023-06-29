import pandas as pd
from pathlib import Path
p_train = Path('join\\train')
p_test = Path('join\\test')


all_train = pd.DataFrame()
all_test = pd.DataFrame()

for file in p_train.glob('*.csv'):
    data=pd.DataFrame()
    data = pd.read_csv(file, delimiter=',', names=['1','2','3','4','5','6'])
    #print(data)
    all_train = pd.concat([all_train, data], ignore_index=True, axis=0)#axis=1)

for file in p_test.glob('*.csv'):
    tdata=pd.DataFrame()
    tdata = pd.read_csv(file, delimiter=',', names=['1','2','3','4','5','6'])
    #print(data)
    all_test = pd.concat([all_test, tdata], ignore_index=True, axis=0)#axis=1)


print(all_test)
all_train.to_csv('GAZP_big_training_set.csv', header=False, index=False, sep=',')
all_test.to_csv('GZAP_big_test_set.csv', header=False, index=False, sep=',')

