import tensorflow as tf
import tensorflow_quantum as tfq

import gym, cirq, sympy
import numpy as np
from functools import reduce
from collections import deque, defaultdict
import matplotlib.pyplot as plt
from cirq.contrib.svg import SVGCircuit
tf.get_logger().setLevel('ERROR')


def one_qubit_rotations(qubit, symbols):
  return [
    cirq.rx(symbols[0])(qubit),
    cirq.ry(symbols[1])(qubit),
    cirq.rz(symbols[2])(qubit)
  ]

def entangling_layer(qubits):
  cz_ops = [cirq.CZ(q0, q1) for q0, q1 in zip( qubits, qubits[1:])]
  cz_ops += ([cirq.CZ(qubits[0], qubits[-1])]) if len(qubits) != 2 else []
  return cz_ops

def generate_circuit(qubits, n_layers):
  """Perpares data re-uploading circuit on `qubits` with `n_layers` layers """
  print("test")
  n_qubits = len(qubits)
  print("N_qubits", n_qubits)

  # Sympy symbols for variational angels
  params = sympy.symbols(f'theta(0:{3*(n_layers+1)*n_qubits})')
  params = np.asarray(params).reshape((n_layers + 1, n_qubits, 3))
  
  # Sympy symbols for encoding angles
  inputs = sympy.symbols(f'x(0:{n_layers})' + f'_(0:{n_qubits})')
  inputs = np.asarray(inputs).reshape((n_layers, n_qubits))

  print("Defined parameters")

  # Define Circuit
  circuit = cirq.Circuit()

  # Initialized circuit 
  print("initialized circuit")
  for l in range(n_layers):
    # Create Variational Layer
    circuit += cirq.Circuit(one_qubit_rotations(q, params[l, i]) for i, q in enumerate(qubits))
    circuit += entangling_layer(qubits)

    # Create encoding layer
    circuit += cirq.Circuit(cirq.rx(inputs[l, i])(q) for i, q in enumerate(qubits))
  
  return circuit, list(params.flat), list(inputs.flat)

# Check procedure

n_qubits, n_layers = 3, 1
qubits = cirq.GridQubit.rect(1, n_qubits)
circuit, parameters, inputs = generate_circuit(qubits, n_layers)

print("parameters:", parameters)
print("inputs = ", inputs)

SVGCircuit(circuit)