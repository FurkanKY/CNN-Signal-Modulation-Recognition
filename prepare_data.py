import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

with open("RML2016.10a_dict.pkl", 'rb') as f:
    data = pickle.load(f, encoding='latin1')

signals = []
labels = []
modulation_types = []

for key in data:
    modulation = key[0]
    if modulation == b'l\xe4\xab\b':
        continue
    
    signals.extend(data[key])
    
    for _ in range(len(data[key])):
        labels.append(modulation)
    
    if modulation not in modulation_types:
        modulation_types.append(modulation)

X = np.array(signals)
Y = np.array(labels)

Y_numeric = np.array([modulation_types.index(mod) for mod in Y])
Y_one_hot = to_categorical(Y_numeric)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y_one_hot, test_size=0.2, random_state=42)

np.savez("prepared_data.npz", 
         X_train=X_train, Y_train=Y_train, 
         X_test=X_test, Y_test=Y_test, 
         modulation_types=modulation_types)
