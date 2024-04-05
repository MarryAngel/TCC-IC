# Importações
import numpy as np
import pandas as pd
import csv

# Dataset que será formatado
data = pd.read_csv('datasets/lung_cancer.csv')
#data.sort_values(by='Result', inplace=True, ascending=True)

print(data)

# Atributos da base de dados
age = pd.DataFrame(data['Age'])
smokes = pd.DataFrame(data['Smokes'])
areaQ = pd.DataFrame(data['AreaQ'])
alkhol = pd.DataFrame(data['Alkhol'])
classe = pd.DataFrame(data['Result'])

ids = []
x1 = []
x2 = []
x3 = []
x4 = []
y = []

for i in range(0,59):
    ids.append("{:.18e}".format(i))
    x1.append("{:.18e}".format(age['Age'][i]))
    x2.append("{:.18e}".format(smokes['Smokes'][i]))
    x3.append("{:.18e}".format(areaQ['AreaQ'][i]))
    x4.append("{:.18e}".format(alkhol['Alkhol'][i]))
    #y.append(classe['Result'][i])
    y.append("{:.18e}".format(classe['Result'][i]))
    
# Criar novo DataFrame
lung_cancer = pd.DataFrame(ids, columns=['id'])
lung_cancer['Age'] = x1
lung_cancer['Smokes'] = x2
lung_cancer['AreaQ'] = x3
lung_cancer['Alkhol'] = x4
lung_cancer['Result'] = y

#ionosphere.sort_values(by='y', inplace=True, ascending=True)

print(lung_cancer)

#specify path for export
path = r'D:\Projetos Computacao\Quantum OPF - Copia\TCC-IC\dados\lungCancer_formatado.txt'

#export DataFrame to text file
with open(path, 'a') as f:
    df_string = lung_cancer.to_string(header=False, index=False)
    f.write(df_string)
    