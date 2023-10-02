from opfython.stream import loader
import random
import numpy as np

# Carregar um arquivo .txt para um array numpy
try:
    txt0 = loader.load_txt("dados/class0.txt")
except Exception as e:
    raise RuntimeError('Error on load txt')

# Verificação do arquivo txt (caso ele volte vazio)
if txt0 is None:
    raise RuntimeError('txt is None')

try:
    txt1 = loader.load_txt("dados/class1.txt")
except Exception as e:
    raise RuntimeError('Error on load txt')

# Verificação do arquivo txt (caso ele volte vazio)
if txt1 is None:
    raise RuntimeError('txt is None')

try:
    txt2 = loader.load_txt("dados/class2.txt")
except Exception as e:
    raise RuntimeError('Error on load txt')

# Verificação do arquivo txt (caso ele volte vazio)
if txt2 is None:
    raise RuntimeError('txt is None')

txt0 = txt0.flatten()

dados0 = np.random.choice(txt0, size=3, replace=False)

print(dados0)

# random.seed(42)
# random.shuffle(txt0)
# dados0 = txt0[:3]

# random.seed(42)
# random.shuffle(txt1)
# dados1 = txt1[:3]

# arquivo = np.concatenate((dados0, dados1))

# print(arquivo)

# Exportar para um arquivo .txt dados1 e dados2

# numpy.savetxt("file2.txt", arquivo)
  
# content = numpy.loadtxt('file2.txt')