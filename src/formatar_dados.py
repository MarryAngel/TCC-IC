# Importações
import numpy as np
import pandas as pd
import csv

# Dataset que será formatado
data = pd.read_csv('datasets/ionosphere.csv')
data.sort_values(by='classe', inplace=True, ascending=True)

print(data)

# Atributos da base de dados
a01 = pd.DataFrame(data['a01'])
a02 = pd.DataFrame(data['a02'])
a03 = pd.DataFrame(data['a03'])
a04 = pd.DataFrame(data['a04'])
a05 = pd.DataFrame(data['a05'])
a06 = pd.DataFrame(data['a06'])
a07 = pd.DataFrame(data['a07'])
a08 = pd.DataFrame(data['a08'])
a09 = pd.DataFrame(data['a09'])
a10 = pd.DataFrame(data['a10'])
a11 = pd.DataFrame(data['a11'])
a12 = pd.DataFrame(data['a12'])
a13 = pd.DataFrame(data['a13'])
a14 = pd.DataFrame(data['a14'])
a15 = pd.DataFrame(data['a15'])
a16 = pd.DataFrame(data['a16'])
a17 = pd.DataFrame(data['a17'])
a18 = pd.DataFrame(data['a18'])
a19 = pd.DataFrame(data['a19'])
a20 = pd.DataFrame(data['a20'])
a21 = pd.DataFrame(data['a21'])
a22 = pd.DataFrame(data['a22'])
a23 = pd.DataFrame(data['a23'])
a24 = pd.DataFrame(data['a24'])
a25 = pd.DataFrame(data['a25'])
a26 = pd.DataFrame(data['a26'])
a27 = pd.DataFrame(data['a27'])
a28 = pd.DataFrame(data['a28'])
a29 = pd.DataFrame(data['a29'])
a30 = pd.DataFrame(data['a30'])
a31 = pd.DataFrame(data['a31'])
a32 = pd.DataFrame(data['a32'])
a33 = pd.DataFrame(data['a33'])
a34 = pd.DataFrame(data['a34'])
classe = pd.DataFrame(data['classe'])

ids = []
x1 = []
x2 = []
x3 = []
x4 = []
x5 = []
x6 = []
x7 = []
x8 = []
x9 = []
x10 = []
x11 = []
x12 = []
x13 = []
x14 = []
x15 = []
x16 = []
x17 = []
x18 = []
x19 = []
x20 = []
x21 = []
x22 = []
x23 = []
x24 = []
x25 = []
x26 = []
x27 = []
x28 = []
x29 = []
x30 = []
x31 = []
x32 = []
x33 = []
x34 = []
y = []

for i in range(0,351):
    ids.append("{:.18e}".format(i))
    x1.append("{:.18e}".format(a01['a01'][i]))
    x2.append("{:.18e}".format(a02['a02'][i]))
    x3.append("{:.18e}".format(a03['a03'][i]))
    x4.append("{:.18e}".format(a04['a04'][i]))
    x5.append("{:.18e}".format(a05['a05'][i]))
    x6.append("{:.18e}".format(a06['a06'][i]))
    x7.append("{:.18e}".format(a07['a07'][i]))
    x8.append("{:.18e}".format(a08['a08'][i]))
    x9.append("{:.18e}".format(a09['a09'][i]))
    x10.append("{:.18e}".format(a10['a10'][i]))
    x11.append("{:.18e}".format(a11['a11'][i]))
    x12.append("{:.18e}".format(a12['a12'][i]))
    x13.append("{:.18e}".format(a13['a13'][i]))
    x14.append("{:.18e}".format(a14['a14'][i]))
    x15.append("{:.18e}".format(a15['a15'][i]))
    x16.append("{:.18e}".format(a16['a16'][i]))
    x17.append("{:.18e}".format(a17['a17'][i]))
    x18.append("{:.18e}".format(a18['a18'][i]))
    x19.append("{:.18e}".format(a19['a19'][i]))
    x20.append("{:.18e}".format(a20['a20'][i]))
    x21.append("{:.18e}".format(a21['a21'][i]))
    x22.append("{:.18e}".format(a22['a22'][i]))
    x23.append("{:.18e}".format(a23['a23'][i]))
    x24.append("{:.18e}".format(a24['a24'][i]))
    x25.append("{:.18e}".format(a25['a25'][i]))
    x26.append("{:.18e}".format(a26['a26'][i]))
    x27.append("{:.18e}".format(a27['a27'][i]))
    x28.append("{:.18e}".format(a28['a28'][i]))
    x29.append("{:.18e}".format(a29['a29'][i]))
    x30.append("{:.18e}".format(a30['a30'][i]))
    x31.append("{:.18e}".format(a31['a31'][i]))
    x32.append("{:.18e}".format(a32['a32'][i]))
    x33.append("{:.18e}".format(a33['a33'][i]))
    x34.append("{:.18e}".format(a34['a34'][i]))
    y.append(classe['classe'][i])
    # y.append("{:.18e}".format(outcome['Outcome'][i]))
    
# Criar novo DataFrame
ionosphere = pd.DataFrame(ids, columns=['id'])
ionosphere['a01'] = x1
ionosphere['a02'] = x2
ionosphere['a03'] = x3
ionosphere['a04'] = x4
ionosphere['a05'] = x5
ionosphere['a06'] = x6
ionosphere['a07'] = x7
ionosphere['a08'] = x8
ionosphere['a09'] = x9
ionosphere['a10'] = x10
ionosphere['a11'] = x11
ionosphere['a12'] = x12
ionosphere['a13'] = x13
ionosphere['a14'] = x14
ionosphere['a15'] = x15
ionosphere['a16'] = x16
ionosphere['a17'] = x17
ionosphere['a18'] = x18
ionosphere['a19'] = x19
ionosphere['a20'] = x20
ionosphere['a21'] = x21
ionosphere['a22'] = x22
ionosphere['a23'] = x23
ionosphere['a24'] = x24
ionosphere['a25'] = x25
ionosphere['a26'] = x26
ionosphere['a27'] = x27
ionosphere['a28'] = x28
ionosphere['a29'] = x29
ionosphere['a30'] = x30
ionosphere['a31'] = x31
ionosphere['a32'] = x32
ionosphere['a33'] = x33
ionosphere['a34'] = x34
ionosphere['y'] = y

ionosphere.sort_values(by='y', inplace=True, ascending=True)

print(ionosphere)

#specify path for export
path = r'D:\Projetos Computacao\Quantum OPF - Copia\TCC-IC\dados\ionosphere_formatado.txt'

#export DataFrame to text file
with open(path, 'a') as f:
    df_string = ionosphere.to_string(header=False, index=False)
    f.write(df_string)
    