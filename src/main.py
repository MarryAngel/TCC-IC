##################################################################################

# Importações opfython (OPF)
import opfython.math.general as g
import opfython.stream.parser as p
import opfython.stream.splitter as s
from opfython.models import SupervisedOPF
from opfython.stream import loader
from qsupervised import QSupervisedOPF

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

# Analisar um array numpy pré-carregado
# X e Y conterão a segunda e terceira coluna do arquivo

X, Y = p.parse_loader(txt)

# Dividir os dados em conjunto de treinamento e conjunto de testes
X_treino, X_teste, Y_treino, Y_teste = s.split(X, Y, percentage=0.5, random_state=1)

# Criar uma instância de SupervisedOPF
#opf = SupervisedOPF(distance="log_squared_euclidean", pre_computed_distance=None)
q_opf = QSupervisedOPF(distance="log_squared_euclidean", pre_computed_distance=None)

# Fits training data into the classifier
#opf.fit(X_treino, Y_treino)
q_opf.fit(X_treino, Y_treino)

# Predicts new data
#preds = opf.predict(X_test)

# Calculating accuracy
#acc = g.opf_accuracy(Y_test, preds)

#print(f"Accuracy: {acc}")