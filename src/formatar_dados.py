# Importações
import numpy as np
import pandas as pd
import csv

# Dataset que será formatado
data = pd.read_csv('datasets/diabetes.csv')
data.sort_values(by='Outcome', inplace=True, ascending=True)

print(data)

# Atributos da base de dados
pregnancies = pd.DataFrame(data['Pregnancies'])
glucose = pd.DataFrame(data['Glucose'])
blood_pressure = pd.DataFrame(data['BloodPressure'])
skin_thickness = pd.DataFrame(data['SkinThickness'])
insulin = pd.DataFrame(data['Insulin'])
bmi = pd.DataFrame(data['BMI'])
diabetes_pedigree_function = pd.DataFrame(data['DiabetesPedigreeFunction'])
age = pd.DataFrame(data['Age'])
outcome = pd.DataFrame(data['Outcome'])
# age = pd.DataFrame(data['age'])
# sex = pd.DataFrame(data['sex'])
# cp = pd.DataFrame(data['cp'])
# trestbps = pd.DataFrame(data['trestbps'])
# chol = pd.DataFrame(data['chol'])
# fbs = pd.DataFrame(data['fbs'])
# restecg = pd.DataFrame(data['restecg'])
# thalach = pd.DataFrame(data['thalach'])
# exang = pd.DataFrame(data['exang'])
# oldpeak = pd.DataFrame(data['oldpeak'])
# slope = pd.DataFrame(data['slope'])
# ca = pd.DataFrame(data['ca'])
# thal = pd.DataFrame(data['thal'])
# target = pd.DataFrame(data['target'])

ids = []
x1 = []
x2 = []
x3 = []
x4 = []
x5 = []
x6 = []
x7 = []
x8 = []
y = []

for i in range(0,768):
    ids.append("{:.18e}".format(i))
    x1.append("{:.18e}".format(pregnancies['Pregnancies'][i]))
    x2.append("{:.18e}".format(glucose['Glucose'][i]))
    x3.append("{:.18e}".format(blood_pressure['BloodPressure'][i]))
    x4.append("{:.18e}".format(skin_thickness['SkinThickness'][i]))
    x5.append("{:.18e}".format(insulin['Insulin'][i]))
    x6.append("{:.18e}".format(bmi['BMI'][i]))
    x7.append("{:.18e}".format(diabetes_pedigree_function['DiabetesPedigreeFunction'][i]))
    x8.append("{:.18e}".format(age['Age'][i]))
    y.append("{:.18e}".format(outcome['Outcome'][i]))
    
# Criar novo DataFrame
diabetes = pd.DataFrame(ids, columns=['id'])
diabetes['x1'] = x1
diabetes['x2'] = x2
diabetes['x3'] = x3
diabetes['x4'] = x4
diabetes['x5'] = x5
diabetes['x6'] = x6
diabetes['x7'] = x7
diabetes['x8'] = x8
diabetes['y'] = y

diabetes.sort_values(by='y', inplace=True, ascending=True)

print(diabetes)

#specify path for export
path = r'D:\Projetos Computacao\Quantum OPF - Copia\TCC-IC\dados\diabetes_formatado.txt'

#export DataFrame to text file
with open(path, 'a') as f:
    df_string = diabetes.to_string(header=False, index=False)
    f.write(df_string)
    