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

algorithm_globals.random_seed = 12345


from qiskit.circuit.library import RXXGate, RYYGate, RZZGate, RXGate

def ry_to_rx(theta):
    qc = QuantumCircuit(1)
    qc.rx(np.pi / 2, 0)
    qc.rx(theta, 0)
    qc.rx(-np.pi / 2, 0)
    return qc

def rz_to_rx(theta):
    qc = QuantumCircuit(1)
    qc.rx(np.pi / 2, 0)
    qc.rx(np.pi / 2, 0)
    qc.rx(theta, 0)
    qc.rx(-np.pi / 2, 0)
    qc.rx(-np.pi / 2, 0)
    return qc

def conv_circuit_equiv(params):
    target = QuantumCircuit(2)

    # Decompose rz(-np.pi / 2) on qubit 1 using RX gates
    target.append(rz_to_rx(-np.pi / 2).to_gate(), [1])
    
    # Decompose cx(1, 0) using RXX gates
    target.rxx(np.pi / 2, 1, 0)
    
    # Decompose rz(params[0]) on qubit 0 using RX gates
    target.append(rz_to_rx(params[0]).to_gate(), [0])
    
    # Decompose ry(params[1]) on qubit 1 using RX gates
    target.append(ry_to_rx(params[1]).to_gate(), [1])
    
    # Decompose cx(0, 1) using RXX gates
    target.rxx(np.pi / 2, 0, 1)
    
    # Decompose ry(params[2]) on qubit 1 using RX gates
    target.append(ry_to_rx(params[2]).to_gate(), [1])
    
    # Decompose cx(1, 0) using RXX gates
    target.rxx(np.pi / 2, 1, 0)
    
    # Decompose rz(np.pi / 2) on qubit 0 using RX gates
    target.append(rz_to_rx(np.pi / 2).to_gate(), [0])
    
    return target


def conv_layer_equiv(num_qubits, param_prefix):
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    qubits = list(range(num_qubits))
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        qc = qc.compose(conv_circuit_equiv(params[param_index : (param_index + 3)]), [q1, q2])
        qc.barrier()
        param_index += 3
    for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
        qc = qc.compose(conv_circuit_equiv(params[param_index : (param_index + 3)]), [q1, q2])
        qc.barrier()
        param_index += 3

    qc_inst = qc.to_instruction()

    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, qubits)
    return qc



def pool_circuit_equiv(params):
    target = QuantumCircuit(2)

    # Decompose rz(-np.pi / 2) on qubit 1 using RX gates
    target.append(rz_to_rx(-np.pi / 2).to_gate(), [1])
    
    # Decompose cx(1, 0) using RXX gates
    target.rxx(np.pi / 2, 1, 0)
    
    # Decompose rz(params[0]) on qubit 0 using RX gates
    target.append(rz_to_rx(params[0]).to_gate(), [0])
    
    # Decompose ry(params[1]) on qubit 1 using RX gates
    target.append(ry_to_rx(params[1]).to_gate(), [1])
    
    # Decompose cx(0, 1) using RXX gates
    target.rxx(np.pi / 2, 0, 1)
    
    # Decompose ry(params[2]) on qubit 1 using RX gates
    target.append(ry_to_rx(params[2]).to_gate(), [1])
    
    return target


def pool_layer_equiv(sources, sinks, param_prefix):
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    for source, sink in zip(sources, sinks):
        qc = qc.compose(pool_circuit_equiv(params[param_index : (param_index + 3)]), [source, sink])
        qc.barrier()
        param_index += 3

    qc_inst = qc.to_instruction()

    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, range(num_qubits))
    return qc

def encode_image_to_quantum(image):
    # Flatten the image to a 1D array
    image = image.flatten()
    
    # Normalize the image pixel values to form a valid quantum state
    normalized_image = normalize(image)

    # print(normalized_image)
    
    # Initialize a quantum circuit with 4 qubits
    num_qubits = 10
    qc = QuantumCircuit(num_qubits)
    
    # Prepare the quantum state from the normalized pixel values
    qc.initialize(normalized_image, range(num_qubits))
    
    return qc
