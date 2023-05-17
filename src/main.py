import pdb
import opfython.math.general as g
import opfython.stream.parser as p
import opfython.stream.splitter as s
from opfython.models import SupervisedOPF
from opfython.stream import loader

# Loading a .txt file to a numpy array
try:
    txt = loader.load_txt("boat.txt")
except Exception as e:
    raise RuntimeError('Error on load txt')

if txt is None:
    raise RuntimeError('txt is None')

print('Arquivo boat carregado')
pdb.set_trace()

# Parsing a pre-loaded numpy array
X, Y = p.parse_loader(txt)


# Splitting data into training and testing sets
## X_train, X_test, Y_train, Y_test = s.split(X, Y, percentage=0.5, random_state=1)

# Creates a SupervisedOPF instance
## opf = SupervisedOPF(distance="log_squared_euclidean", pre_computed_distance=None)

# Fits training data into the classifier
## opf.fit(X_train, Y_train)

# Predicts new data
##preds = opf.predict(X_test)

# Calculating accuracy
##acc = g.opf_accuracy(Y_test, preds)

##print(f"Accuracy: {acc}")