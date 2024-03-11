import opfython.math.general as g
import opfython.stream.parser as p
import opfython.stream.splitter as s
from opfython.models import SupervisedOPF
from opfython.stream import loader
from qsupervised import QSupervisedOPF
import numpy as np

# Outras importações
import pdb #biblioteca para debugar
# pdb.set_trace() # comando para debugar

##################################################################################

# Carregar um arquivo .txt para um array numpy
try:
    txt = loader.load_txt("dados/boat.txt")
except Exception as e:
    raise RuntimeError('Error on load txt')

# Verificação do arquivo txt (caso ele volte vazio)
if txt is None:
    raise RuntimeError('txt is None')

# From third columns beyond, we should have the features
X = txt[:, 2:]

# Second column should be the label
Y = txt[:, 1]
Y = Y.astype(int)


# Analisar um array numpy pré-carregado
# X e Y conterão a segunda e terceira coluna do arquivo

#X, Y = p.parse_loader(txt)

# ------------------------------------------------------------

# Escolher duas classes aleatórias com uma quantidade x de amostras cada uma

num_classes = 2
num_samples_per_class = 4

# Get the unique classes from Y and their respective indices
unique_classes = np.unique(Y)

# Randomly choose two classes in order 
random_classes = np.random.choice(unique_classes, size=num_classes, replace=False)

# Ordenar random_classes
random_classes = np.sort(random_classes)

# Initialize empty lists to store the selected samples
selected_X = []
selected_Y = []
label = 0

# Iterate over each class
for class_label in random_classes:
    # Get the indices of samples belonging to the current class
    class_indices = np.where(Y == class_label)[0]
    
    # Randomly select num_samples_per_class samples from the current class
    selected_indices = np.random.choice(class_indices, size=num_samples_per_class, replace=False)
    
    # Append the selected samples to the selected_X and selected_Y lists
    selected_X.extend(X[selected_indices])
    selected_Y.extend(Y[selected_indices])
    
# Renumerar as classes começando em 0 até num_classes-1
for i in range(len(selected_Y)):
    if selected_Y[i] == random_classes[0]:
        selected_Y[i] = 0
    else:
        selected_Y[i] = 1
    
# Convert the selected_X and selected_Y lists to numpy arrays
selected_X = np.array(selected_X)
selected_Y = np.array(selected_Y)

# ------------------------------------------------------------

# Use the selected samples for training and testing
X_treino, X_teste, Y_treino, Y_teste = s.split(selected_X, selected_Y, percentage=0.5, random_state=1)

# Dividir os dados em conjunto de treinamento e conjunto de testes
#X_treino, X_teste, Y_treino, Y_teste = s.split(X, Y, percentage=0.5, random_state=1)

# Criar uma instância de SupervisedOPF
opf = SupervisedOPF(distance="log_squared_euclidean", pre_computed_distance=None)

# Fits training data into the classifier
opf.fit(X_treino, Y_treino)

# Predicts new data
preds = opf.predict(X_teste)

# Calculating accuracy
acc = g.opf_accuracy(Y_teste, preds)

print(f"Accuracy: {acc}")

q_opf = QSupervisedOPF(distance="log_squared_euclidean", pre_computed_distance=None)

# Fits training data into the classifier
q_opf.fit(X_treino, Y_treino)

# Predicts new data
preds_q = q_opf.predict(X_teste)

# Calculating accuracy
acc_q = g.opf_accuracy(Y_teste, preds_q)

print(f"Accuracy Q: {acc_q}")

