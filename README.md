# Expert-System
Classification of Lymphoblastic Leukaemia using Quantum Neural Network
Step#1: Data Augmentation in term Horizontal and Vertical Flip
Step#2: Import Libraries 
[ import pennylane as qml
from pennylane import numpy as np]
The pip install pennylane command is used to install the PennyLane package in Python. PennyLane is a library for quantum machine learning, quantum computing, and quantum chemistry. By running this command in your terminal or coding environment, you download and set up PennyLane so you can use it to develop and run quantum algorithms, create quantum neural networks, and simulate quantum systems.
Hyperparameters [In this setup, we have a quantum machine learning model configuration defined by several key parameters].
	The variable n-qubits = 4 sets the number of qubits used in the quantum circuit, establishing the model's quantum state space.
	The step = 0.0008 defines the learning rate, controlling the adjustment rate for the model's parameters during optimization.
	The batch-size = 16 specifies the number of samples processed per training step, while num-epochs = 100 sets the total training iterations to 100. 
	Circuit complexity is handled by q-depth = 8, indicating the number of variational layers in the quantum circuit.
	The learning rate scheduler gamma lr-scheduler = 0.2 reduces the learning rate every 10 epochs by a factor of 0.2 to aid convergence. 
	For initializing quantum weights, q-delta = 0.002 provides a small spread to start the parameters randomly. 
	Finally, start-time = time. time () records the computation’s start time to measure total execution duration.

IMPORT Lymphoblastic Leukaemia DATASET
Hadamard gates
These functions define essential layers in a quantum neural network for creating and manipulating qubit states. H-layer(nqubits): 
This function applies a layer of Hadamard gates, qml.Hadamard(wires=idx), to each qubit in the system. 
The Hadamard gate places each qubit in a superposition, allowing for interference and entanglement in later steps.

RY_layer(w): This function applies a rotation around the Y-axis for each qubit, using angles from the parameter vector w. Each qml.RY(element, wires=idx) rotation is an adjustable operation, enabling the model to learn complex patterns through parameterized rotations.

Entangling_layer(nqubits): This layer creates entanglement between adjacent qubits using controlled NOT (CNOT) gates. First, it applies CNOT gates between even-indexed qubits (e.g., qubit 0 and 1, 2 and 3), then applies a shifted layer of CNOT gates on odd indices (e.g., qubit 1 and 2, 3 and 4). This alternation ensures robust entanglement across the circuit, allowing qubits to share information. These layers are foundational in constructing a variational quantum circuit for quantum machine learning.
Variational Quantum Circuit
	This function, quantum-net, defines a quantum neural network (QNN) that processes input features and learns parameters for quantum machine learning tasks. It begins by reshaping the flat weight vector q-weights-flat into a matrix q-weights with dimensions based on the circuit depth (q-depth) and number of qubits (n-qubits). 
	The circuit is initialized with a Hadamard layer (H-layer), which places each qubit in an equal superposition state, unbiased between the |0⟩ and |1⟩ states. 
	Then, the input features (q-input-features) are embedded as rotations around the Y-axis via RY-layer, encoding the data directly into the quantum state. 
	The main body of the model consists of q-depth variational layers, each of which applies entangling CNOT gates (entangling-layer) to create dependencies between qubits, followed by a layer of Y-rotations parameterized by q-weights[k], allowing for tunable transformations. 
	Finally, the circuit returns the expectation values of each qubit in the Z-basis using qml.expval(qml.PauliZ(position)), which represents the measurement outputs of the network and serves as the QNN’s output.
Dressed Quantum Net
	The Dressed Quantum Net class combines classical neural network layers with a quantum neural network (QNN), creating a "dressed" hybrid model that leverages the strengths of both quantum and classical computation. In its init method, the model initializes three main components: pre-net, q-params, and post-net.
	The pre-net is a classical linear layer that reduces the input features from 512 to n-qubits, preparing the data for the quantum circuit. 
	The q-params holds trainable quantum parameters, initialized with small random values based on q-delta to set up the variational layers of the QNN. 
	Finally, post-net is another linear layer that maps the output of the QNN to a two-dimensional prediction space, suitable for classification.
	The forward method defines the data flow through the model.
	First, the high-dimensional input features are passed through pre-net and then transformed with a tanh activation scaled by π/2, yielding the values required for quantum rotations. 
	These transformed features, q-in, are then fed in batches to the quantum circuit quantum-net, with each batch element processed sequentially to generate q-out, the quantum circuit’s output. 
	This quantum-enhanced output is finally passed through post-net to produce a two-dimensional prediction, integrating quantum computation into the classical neural network pipeline.
ResNet18 initializes a pre-trained ResNet-18 model using weights from the ImageNet dataset, adapting it to incorporate a quantum-classical hybrid layer for specialized processing. 

By setting param. requires-grad = False for each parameter in model-hybrid, the pre-trained layers are frozen, meaning their weights will not be updated during training. This preserves the learned features of ResNet-18, effectively using it as a feature extractor. The final layer (model-hybrid.fc) is replaced with DressedQuantumNet(), a custom module that includes both classical and quantum components, allowing the model to make predictions based on quantum-enhanced representations. 
Finally, model-hybrid is transferred to either a CUDA or CPU device, depending on the available hardware, ensuring compatibility with the rest of the training setup. This hybrid model leverages both deep learning's strengths in feature extraction and quantum computation's potential in representation learning.
CrossEntropyLoss
Model Training
# Save the trained model
Model-path = 'model-hybrid.pth'  # Specify the path where you want to save the model
torch.save(model-hybrid.state_dict(), model_path)
print(f'Model saved to {model-path}')
Prediction Results based on Trained Model

