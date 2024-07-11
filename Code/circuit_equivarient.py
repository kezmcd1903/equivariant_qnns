import json
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.utils import algorithm_globals
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.neural_networks import EstimatorQNN
from sklearn.model_selection import train_test_split

from matplotlib import pyplot
from keras.datasets import fashion_mnist
import numpy as np
from qiskit.circuit import ParameterVector

algorithm_globals.random_seed = 12345


from qiskit.circuit.library import RXXGate, RYYGate, RZZGate, RXGate




def eqv_circ(params):

    qc = QuantumCircuit(8)
    
    # Type A - within 0-3
    qc.rx(-np.pi / 2, 1)
    qc.rxx(np.pi/2, 1,0)
    qc.rx(params[0], 0)
    qc.rx(params[1], 1)
    qc.rxx(np.pi/2, 0,1) ##### can change some of these rxx to ryy or rzz
    qc.rx(params[2], 0)
    qc.rxx(np.pi/2, 1,0)
    qc.rx(np.pi / 2, 0)
    
    # Type A - within 0-3
    qc.rx(-np.pi / 2, 3)
    qc.ryy(np.pi/2, 3,2)
    qc.rx(params[3], 2)
    qc.rx(params[4], 3)
    qc.ryy(np.pi/2, 2,3)
    qc.rx(params[5], 2)
    qc.ryy(np.pi/2, 3,2)   ##### can change some of these rxx to ryy or rzz
    qc.rx(np.pi / 2, 2)
    
    # Type C - across 0-3 and 4-7
    qc.rz(- np.pi/2,5)
    qc.cx(5,4)
    qc.rx(params[6], 4)
    qc.ry(params[7], 5)
    qc.rxx(np.pi/2, 4,5)
    qc.ry(params[8], 5)
    qc.cx(5,4)
    qc.rx(np.pi/2,4)
    
    # Type B - within 4-7
    qc.rz(-np.pi/2,7)
    qc.cx(7,6)
    qc.rz(params[9], 6)
    qc.ry(params[10], 7)
    qc.cx(6,7)
    qc.ry(params[11], 7)
    qc.cx(7,6)
    qc.rz(np.pi/2,6)
    
    # Type A - within 0-3
    qc.rx(-np.pi / 2, 3)
    qc.rzz(np.pi/2, 3,2)
    qc.rx(params[12], 2)
    qc.rx(params[13], 3)
    qc.rzz(np.pi/2, 2,3)
    qc.rx(params[14], 2)
    qc.rzz(np.pi/2, 3,2)   ##### can change some of these rxx to ryy or rzz
    qc.rx(np.pi / 2, 2)
    
    
    qc.rx(-np.pi / 2, 4)
    qc.rxx(np.pi/2, 4,3)
    qc.rx(params[15], 3)
    qc.rx(params[16], 4)
    qc.rxx(np.pi/2, 3,4)
    qc.rx(params[17], 3)
    qc.rxx(np.pi/2, 4,3)   ##### 
    qc.rx(np.pi / 2, 3)
    
    
    
    qc.rz(-np.pi/2,6)
    qc.cx(6,5)
    qc.rz(params[18], 5)
    qc.ry(params[19], 6)
    qc.cx(5,6)
    qc.ry(params[20], 6)
    qc.cx(6,5)
    qc.rz(np.pi/2,5)
    
    
    # longer entangling
    qc.rx(-np.pi / 2, 0)
    qc.rxx(np.pi/2, 0,7)
    qc.rx(params[21], 0)
    qc.rz(params[22], 7)
    qc.cx(7,0)
    qc.rx(params[23], 0)
    qc.rxx(np.pi/2, 0,7)
    qc.rz(np.pi/2,7)
    
    
    # longer entangling
    qc.rz(-np.pi / 2, 4)
    qc.cx(4,0)
    qc.rx(params[24], 0)
    qc.ry(params[25], 4)
    qc.rxx(np.pi/2,0,4)
    qc.ry(params[26],4)
    
    
    # longer entangling
    qc.rz(-np.pi / 2, 5)
    qc.cx(5,1)
    qc.rx(params[27], 1)
    qc.ry(params[28], 5)
    qc.rxx(np.pi/2,1,5)
    qc.ry(params[29],5)
    
    
    # longer entangling
    qc.rz(-np.pi / 2, 6)
    qc.cx(6,2)
    qc.rx(params[30], 2)
    qc.ry(params[31], 6)
    qc.rxx(np.pi/2,2,6)
    qc.ry(params[32],6)
    
    
    # longer entangling
    qc.rz(-np.pi / 2, 7)
    qc.cx(7,3)
    qc.rx(params[33], 3)
    qc.ry(params[34], 7)
    qc.rxx(np.pi/2,3,7)
    qc.ry(params[35],7)
    
    
    # Type A - within 0-3
    qc.rx(-np.pi / 2, 1)
    qc.rxx(np.pi/2, 1,0)
    qc.rx(params[36], 0)
    qc.rx(params[37], 1)
    qc.rxx(np.pi/2, 0,1) ##### can change some of these rxx to ryy or rzz
    qc.rx(params[38], 0)
    qc.rxx(np.pi/2, 1,0)
    qc.rx(np.pi / 2, 0)
    
    # Type A - within 0-3
    qc.rx(-np.pi / 2, 3)
    qc.ryy(np.pi/2, 3,2)
    qc.rx(params[39], 2)
    qc.rx(params[40], 3)
    qc.ryy(np.pi/2, 2,3)
    qc.rx(params[41], 2)
    qc.ryy(np.pi/2, 3,2)   ##### can change some of these rxx to ryy or rzz
    qc.rx(np.pi / 2, 2)
    
    
    # Type A - within 0-3
    qc.rx(-np.pi / 2, 2)
    qc.rzz(np.pi/2, 2,1)
    qc.rx(params[42], 1)
    qc.rx(params[43], 2)
    qc.rzz(np.pi/2, 1,2)
    qc.rx(params[44], 1)
    qc.rzz(np.pi/2, 2,1)   ##### can change some of these rxx to ryy or rzz
    qc.rx(np.pi / 2, 1)    

    
    # Type A - within 0-3
    qc.rx(-np.pi / 2, 0)
    qc.rxx(np.pi/2, 0,3)
    qc.rx(params[45], 0)
    qc.rx(params[46], 3)
    qc.rxx(np.pi/2, 3,0) ##### can change some of these rxx to ryy or rzz
    qc.rx(params[47], 0)
    qc.rxx(np.pi/2, 0,3)
    qc.rx(np.pi / 2, 3)
    
    
    #  within 0-3
    qc.rx(-np.pi / 2, 2)
    qc.ryy(np.pi/2, 2,0)
    qc.rx(params[48], 0)
    qc.rx(params[49], 2)
    qc.ryy(np.pi/2, 0,2) ##### can change some of these rxx to ryy or rzz
    qc.rx(params[50], 2)
    
    
    #  within 0-3
    qc.rx(-np.pi / 2, 3)
    qc.rzz(np.pi/2, 3,1)
    qc.rx(params[51], 1)
    qc.rx(params[52], 3)
    qc.rzz(np.pi/2, 1,3) ##### can change some of these rxx to ryy or rzz
    qc.rx(params[53], 3)
    
    
    # Type A - within 0-3
    qc.rx(-np.pi / 2, 1)
    qc.rxx(np.pi/2, 1,0)
    qc.rx(params[54], 0)
    qc.rx(params[55], 1)
    qc.rxx(np.pi/2, 0,1) ##### can change some of these rxx to ryy or rzz
    qc.rx(params[56], 0)
    qc.rxx(np.pi/2, 1,0)
    qc.rx(np.pi / 2, 0)
    
    
    # Type A - within 0-3
    qc.rx(-np.pi / 2, 0)
    qc.ryy(np.pi/2, 0,1)
    qc.rx(params[57], 0)
    qc.rx(params[58], 1)
    qc.ryy(np.pi/2, 1,0) ##### can change some of these rxx to ryy or rzz
    qc.rx(params[59], 0)
    qc.ryy(np.pi/2, 0,1)
    qc.rx(np.pi / 2, 1)
    
    
    # Type A - within 0-3
    qc.rx(-np.pi / 2, 1)
    qc.rzz(np.pi/2, 1,0)
    qc.rx(params[60], 0)
    qc.rx(params[61], 1)
    qc.rzz(np.pi/2, 0,1) ##### can change some of these rxx to ryy or rzz
    qc.rx(params[62], 0)
    qc.rzz(np.pi/2, 1,0)
    qc.rx(np.pi / 2, 0)

    return qc
    
