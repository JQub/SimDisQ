import numpy as np
from qiskit.providers import BackendV2, Options
from qiskit.transpiler import Target, InstructionProperties
from qiskit.circuit.library import XGate, SXGate, RZGate, CZGate
from qiskit.circuit import Measure, Delay, Parameter, Reset
from qiskit import QuantumCircuit, transpile
import matplotlib.pyplot as plt
from qiskit_aer.noise import NoiseModel
from qiskit_aer import AerSimulator
from qiskit.providers.backend import QubitProperties

class IonQ(BackendV2):
    """Fake Aria-2 backend with full connectivity."""
    
    def __init__(self):
        """Instantiate a new fake Aria-2 backend.
        
        This backend simulates a 25-qubit fully connected quantum processor
        based on the qpu.aria-2 specifications.
        """
        super().__init__(name="IonQ backend")
        
        # Backend properties from the provided spec
        self._num_qubits = 25
        self.backend_name = "qpu.IonQ"
        
        # Fidelity parameters
        self.fidelity_1q_mean = 0.9997
        self.fidelity_2q_mean = 0.9699
        self.fidelity_spam_mean = 0.9974
        
        # Timing parameters (in seconds)
        self.t_readout = 0.00005  # 50 us
        self.t_reset = 0.000015   # 15 us
        self.t_1q = 0.000135      # 135 us
        self.t_2q = 0.0006        # 600 us
        self.t1_time = 10         # 10 s
        self.t2_time = 1.5        # 1.5 s
        
        # Set up random number generator with seed for reproducibility
        rng = np.random.default_rng(seed=12345678942)
        
        # Create qubit properties for thermal relaxation errors
        qubit_properties = []
        for i in range(self._num_qubits):
            # Add some variation to T1 and T2 times
            t1_var = rng.uniform(-0.5, 0.5)  # +/-0.5 s variation
            t2_var = rng.uniform(-0.1, 0.1)  # +/-0.1 s variation
            
            qubit_properties.append(
                QubitProperties(
                    t1=self.t1_time + t1_var,
                    t2=self.t2_time + t2_var,
                    frequency=rng.uniform(4.5e9, 5.5e9)  # 4.5-5.5 GHz
                )
            )
        
        # Create target with qubit properties
        self._target = Target(
            "IonQ backend",
            num_qubits=self._num_qubits,
            qubit_properties=qubit_properties
        )
        
        # Single qubit gate properties
        rz_props = {}
        x_props = {}
        sx_props = {}
        measure_props = {}
        delay_props = {}
        reset_props = {}
        
        # Add 1q gates for all qubits
        for i in range(self._num_qubits):
            qarg = (i,)
            
            # RZ is virtual (no error, no duration)
            rz_props[qarg] = InstructionProperties(error=0.0, duration=0.0)
            
            # X gate - error derived from 1q fidelity
            error_1q = 1 - self.fidelity_1q_mean
            x_props[qarg] = InstructionProperties(
                error=error_1q + rng.uniform(-1e-5, 1e-5),
                duration=self.t_1q,
            )
            
            # SX gate - similar error to X
            sx_props[qarg] = InstructionProperties(
                error=error_1q + rng.uniform(-1e-5, 1e-5),
                duration=self.t_1q,
            )
            
            # Measurement - error from SPAM fidelity
            error_spam = 1 - self.fidelity_spam_mean
            measure_props[qarg] = InstructionProperties(
                error=error_spam + rng.uniform(-1e-4, 1e-4),
                duration=self.t_readout,
            )
            
            # Reset
            reset_props[qarg] = InstructionProperties(
                error=error_spam + rng.uniform(-1e-4, 1e-4),
                duration=self.t_reset,
            )
            
            # Delay (no error)
            delay_props[qarg] = None
        
        # Add single qubit instructions to target
        self._target.add_instruction(XGate(), x_props)
        self._target.add_instruction(SXGate(), sx_props)
        self._target.add_instruction(RZGate(Parameter("theta")), rz_props)
        self._target.add_instruction(Measure(), measure_props)
        self._target.add_instruction(Reset(), reset_props)
        self._target.add_instruction(Delay(Parameter("t")), delay_props)
        
        # Two qubit gate properties - Full connectivity with CZ
        cz_props = {}
        error_2q = 1 - self.fidelity_2q_mean
        
        # Create all possible qubit pairs (full connectivity)
        for i in range(self._num_qubits):
            for j in range(i + 1, self._num_qubits):
                # Add both directions for bidirectional connectivity
                edge_forward = (i, j)
                edge_backward = (j, i)
                
                # Add some variation to 2q gate errors
                error_var = rng.uniform(-5e-3, 5e-3)
                
                cz_props[edge_forward] = InstructionProperties(
                    error=error_2q + error_var,
                    duration=self.t_2q,
                )
                cz_props[edge_backward] = InstructionProperties(
                    error=error_2q + error_var,
                    duration=self.t_2q,
                )
        
        self._target.add_instruction(CZGate(), cz_props)
    
    @property
    def target(self):
        return self._target
    
    @property
    def max_circuits(self):
        return None
    
    @property
    def num_qubits(self):
        return self._num_qubits
    
    @classmethod
    def _default_options(cls):
        return Options(shots=1024)
    
    def run(self, circuit, **kwargs):
        raise NotImplementedError(
            "This backend does not contain a run method"
        )
