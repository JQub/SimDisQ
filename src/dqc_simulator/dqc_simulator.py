from qiskit import QuantumCircuit, ClassicalRegister, transpile
from qiskit.circuit import Instruction, CircuitInstruction
from qiskit_aer import AerSimulator
from qiskit.circuit.library.standard_gates import HGate, XGate, ZGate, CXGate
from qiskit.circuit import Reset, Measure, ClassicalRegister
from qiskit.visualization import plot_histogram
from qiskit import QuantumRegister
from qiskit.quantum_info import Kraus
from qiskit.converters import circuit_to_dag
from qiskit_ibm_runtime.fake_provider import (
    FakeVigoV2,         # 5
    FakeLagosV2,        # 7
    FakeCasablancaV2,
    FakeYorktownV2,
    FakeManilaV2,
    FakeNairobiV2,
    FakeMumbaiV2,
    FakeKolkataV2,
    FakeGuadalupeV2,
    FakeAlmadenV2,
    FakeAthensV2,       # 5
    FakeCambridgeV2                
)

# Backend IonQ
from .backend import IonQ

# Timing interface and scheduler
from .timing import DQCTimingProvider, QiskitTimingProvider, NetworkTimingProvider
from .scheduler import DQCScheduler
from .decoherence import DecoherenceModel, add_idle_noise

from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error, pauli_error,
    amplitude_damping_error, phase_amplitude_damping_error,
    phase_damping_error
)

import numpy as np
import copy  
import matplotlib.pyplot as plt

class RemoteGate(Instruction):
    """ Remote Gate"""
    def __init__(self, index: int, target: int):
        super().__init__("R", 1, 0, [])
        self.index = index
        self.target = target

class MX(Instruction):
    """MX for Control Side"""
    def __init__(self, index: int, target: int):
        super().__init__("MX", 2, 0, [])
        self.index = index
        self.target = target

class MZ(Instruction):
    """MZ for Target Side"""
    def __init__(self, index: int, target: int):
        super().__init__("MZ", 2, 0, [])
        self.index = index
        self.target = target

class AnsM(Instruction):
    """AnsM Measurement"""
    def __init__(self, mea: int):
        super().__init__("ANS_M", 1, 0, [])
        self.mea = mea

class S_CX(Instruction):
    """Custom CNOT Gate"""
    def __init__(self, control: int, target: int, path):
        super().__init__("S_CX", 2, 0, [])
        self.control = control
        self.target = target
        self.path = path

class MS(Instruction):
    """Multi-qubit Swap Gate"""
    def __init__(self, index: int, target: int):
        super().__init__("MS", 2, 0, [])
        self.index = index
        self.target = target

class IF_Z(Instruction):
    def __init__(self, index: int, target: int):
        super().__init__("IF_Z", 1, 0, [])
        self.index = index
        self.target = target

class IF_X(Instruction):
    """Conditional X Gate"""
    def __init__(self, index: int, target: int):
        super().__init__("IF_X", 1, 0, [])
        self.index = index
        self.target = target

# Mapping backend names to their classes
FAKE_BACKENDS = {
    "FakeVigoV2": FakeVigoV2,
    "FakeLagosV2": FakeLagosV2,
    "FakeCasablancaV2": FakeCasablancaV2,
    "FakeYorktownV2": FakeYorktownV2,
    "FakeManilaV2": FakeManilaV2,
    "FakeNairobiV2": FakeNairobiV2,
    "FakeMumbaiV2": FakeMumbaiV2,
    "FakeKolkataV2": FakeKolkataV2,
    "FakeGuadalupeV2": FakeGuadalupeV2,
    "FakeAlmadenV2": FakeAlmadenV2,
    "FakeAthensV2": FakeAthensV2,
    "FakeCambridgeV2": FakeCambridgeV2,
    "IonQ": IonQ
}

class QPUManager:
    def __init__(self, timing=None):
        """
        Initialize QPU manager to handle multiple QPUs and their connections.
        
        Attributes:
            qpus: List of QPU instances
            noise_instructions: Dict mapping QPU pairs to noise instructions
            map: Adjacency list representing QPU network topology
            size: Total number of QPUs in the manager
            timing: TimingProvider (default: Qiskit local timing + default network timing)
        """
        # Store QPUs and noise instructions
        self.qpus = []
        self.noise_instructions = {}
        # Adjacency list: {qpu_id: [(neighbor_id, distance), ...]}
        self.map = {}  
        self.size = 0

        if timing is None:
            timing = DQCTimingProvider(QiskitTimingProvider(qpu_resolver=self.get_qpu), NetworkTimingProvider())
        self.timing = timing

    def add_qpu(self, qpu):
        """Add a QPU to the manager"""
        self.qpus.append(qpu)
        qpu_id = qpu.qpu_id
        self.size += 1
        # Initialize adjacency list entry if not exists
        if qpu_id not in self.map:
            self.map[qpu_id] = []

    def get_qpu(self, qpu_id):
        """Retrieve QPU instance by ID"""
        for qpu in self.qpus:
            if qpu.qpu_id == qpu_id:
                return qpu
        return None
    
    def add_coonnection(self, qpu_id1, qpu_id2, distance: float = 0,
                        epr_time: float = None, classical_latency: float = None,
                        epr_transmission_time: float = None):
        """
        Add bidirectional connection between two QPUs.

        :param epr_time: EPR generation time on this link in seconds (None -> timing default)
        :param classical_latency: classical message latency in seconds (None -> timing default)
        :param epr_transmission_time: EPR delivery time in seconds (None -> fiber propagation of distance km)
        """
        qpu1 = self.get_qpu(qpu_id1)
        qpu2 = self.get_qpu(qpu_id2)
        if qpu1 is None or qpu2 is None:
            raise ValueError(f"QPU {qpu_id1} or {qpu_id2} not found.")

        qpu1.add_connection(qpu_id2, distance)
        qpu2.add_connection(qpu_id1, distance)

        # Add to adjacency list if not already present
        if not any(n == qpu_id2 for n, _ in self.map[qpu_id1]):
            self.map[qpu_id1].append((qpu_id2, distance))
        if not any(n == qpu_id1 for n, _ in self.map[qpu_id2]):
            self.map[qpu_id2].append((qpu_id1, distance))

        # Store noise instructions for both directions
        noise_instr = qpu1.get_noise_by_distance(distance)
        self.noise_instructions[(qpu_id1, qpu_id2)] = noise_instr
        self.noise_instructions[(qpu_id2, qpu_id1)] = noise_instr

        # Register link timing
        self.timing.register_link(qpu_id1, qpu_id2, distance=distance,
                                  epr_time=epr_time, classical_latency=classical_latency,
                                  epr_transmission_time=epr_transmission_time)

    def get_noise_instruction(self, qpu_id1, qpu_id2):
        """Get noise instruction for connection between two QPUs"""
        return self.noise_instructions.get((qpu_id1, qpu_id2), None)

    def check_connection(self, qpu_id1, qpu_id2) -> int:
        """Check if two QPUs are connected (returns 1 if connected, 0 otherwise)"""
        if qpu_id1 not in self.map:
            return 0
        return 1 if any(n == qpu_id2 for n, _ in self.map[qpu_id1]) else 0

    # ---- Timing interface (all values in seconds) ----
    def get_gate_time(self, qpu_id, gate, qubits):
        """Local gate time of `gate` on physical `qubits` of QPU `qpu_id`"""
        return self.timing.get_gate_time(self.get_qpu(qpu_id), gate, qubits)

    def get_epr_time(self, qpu_id1, qpu_id2):
        """EPR generation time between two linked QPUs"""
        return self.timing.get_epr_time(qpu_id1, qpu_id2)

    def get_epr_transmission_time(self, qpu_id1, qpu_id2):
        """EPR transmission (delivery) time between two linked QPUs"""
        return self.timing.get_epr_transmission_time(qpu_id1, qpu_id2)

    def get_classical_latency(self, qpu_id1, qpu_id2):
        """Classical communication latency from qpu_id1 to qpu_id2"""
        return self.timing.get_classical_latency(qpu_id1, qpu_id2)

    def get_operation_duration(self, operation, resource, context=None):
        """Unified entry: context is a qpu_id for local gates, None for network events"""
        if isinstance(context, (int, np.integer)):
            context = self.get_qpu(context)
        return self.timing.get_operation_duration(operation, resource, context)


class DQCQPU:
    def __init__(self, qpu_id: int, backend_name: str, connections=None, noise_type: str = None, **noise_kwargs):
        """
        Initialize a QPU instance with automatic custom instruction registration.

        :param qpu_id: QPU identifier
        :param backend_name: Name of the backend (e.g., "FakeLagosV2")
        :param connections: Optional list of connections to other QPUs
        :param noise_type: Type of noise model to apply
        :param noise_kwargs: Additional noise parameters
        """
        self.qpu_id = qpu_id

        # Validate backend name
        if backend_name not in FAKE_BACKENDS:
            raise ValueError(
                f"Unknown backend name '{backend_name}'. "
                f"Available options are: {list(FAKE_BACKENDS.keys())}"
            )

        # Initialize backend
        backend_cls = FAKE_BACKENDS[backend_name]
        backend = backend_cls()

        # Register custom instructions to target
        target = backend.target
        target.add_instruction(RemoteGate, name="R")
        target.add_instruction(MX, name="MX")
        target.add_instruction(MZ, name="MZ")
        target.add_instruction(AnsM, name="ANS_M")
        target.add_instruction(AnsM, name="IF_Z")
        target.add_instruction(AnsM, name="IF_X")
        target.add_instruction(S_CX, name="MS")

        self.backend = backend
        self.target = backend.target

        # Initialize connections and noise configuration
        self.connections = connections if connections else []
        
        if noise_type is not None:
            self.noise_config = {"type": noise_type, "params": noise_kwargs}
        else:
            self.noise_config = None

    def add_connection(self, other_qpu_id: int, distance: float = 1.0):
        """Add connection to another QPU with distance-based noise"""
        kraus = self.get_noise_by_distance(distance)
        conn = {
            "id": other_qpu_id,
            "distance": distance,
            "noise": kraus
        }
        self.connections.append(conn)

    def compile_x_gate(self):
        """Compile X gate for this QPU's backend"""
        qc = QuantumCircuit(1)
        qc.x(0)
        compiled = transpile(qc, self.backend)
        return compiled.data

    def compile_z_gate(self):
        """Compile Z gate for this QPU's backend"""
        qc = QuantumCircuit(1)
        qc.z(0)
        compiled = transpile(qc, self.backend)
        return compiled.data
    
    def exponential(self, L, alpha):
        """Calculate exponential decay based on distance"""
        return np.exp(-alpha * L)

    def get_noise_by_distance(self, distance=1):
        """
        Generate composite noise (amplitude damping + depolarizing) Kraus instruction
        based on transmission distance.
        :param distance: Transmission distance (e.g., km or m)
        :return: Qiskit Instruction containing Kraus operators
        """
        
        # Calculate amplitude damping parameter
        if distance == 0:
            gamma = 0
        else:
            gamma = self.exponential(distance, alpha=0.02)

        # Calculate depolarizing parameter
        p_depol = gamma

        gamma = 1 - gamma
        p_depol = 1 - p_depol

        # Amplitude damping Kraus operators
        K0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex)
        K1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)
        K_amp = [K0, K1]

        # Depolarizing Kraus operators
        sqrt1mp = np.sqrt(1 - p_depol)
        sqrt_p3 = np.sqrt(p_depol / 3)
        I = np.eye(2, dtype=complex)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        K_depol = [sqrt1mp * I, sqrt_p3 * X, sqrt_p3 * Y, sqrt_p3 * Z]

        # Combine channels in order: amplitude damping, then depolarizing.
        K_combined = [D @ A for D in K_depol for A in K_amp]

        # Create Qiskit Kraus instruction
        kraus_instr = Kraus(K_combined).to_instruction()
        
        return kraus_instr

    def __repr__(self):
        return (
            f"<QPU id={self.qpu_id}, size={self.size}, "
            f"backend={self.backend.name()}, max_qubits={self.backend.num_qubits}>"
        )

# -------------------------------------------------------------
class DQCCircuit(QuantumCircuit):
    def __init__(self, *args, **kwargs):
        # Initialize from an existing QuantumCircuit.
        if len(args) == 1 and isinstance(args[0], QuantumCircuit):
            qc = args[0]
            # Preserve the original register structure.
            super().__init__(*qc.qregs, *qc.cregs,
                             name=qc.name,
                             global_phase=qc.global_phase,
                             metadata=copy.deepcopy(qc.metadata) if qc.metadata else None)

            # Copy circuit data.
            self.data = copy.deepcopy(qc.data)
        else:
            super().__init__(*args, **kwargs)

        self.step = []                              # Circuit snapshots for each stage
        self.step.append(copy.deepcopy(self))

        self.sub_circuit = []                       # Sub-circuits
        self.sub_circuit_trans = []                 # Transpiled sub-circuits
        self.result_circuit = None                  # Final merged circuit

        self.partition = []                         # Circuit partition
        self.qubit_group = [-1] * self.num_qubits   # Qubit-to-group mapping
        self.Entanglement_swapping = []             # Entanglement swapping flags
        self.swap_routes = []                       # Swap routes

        self.qubit_tele = []                        # Communication qubit index for each qubit

        self.qpus = []
        self.merged_qubits_map = {}                 # Merged global-to-local qubit map
        self.merged_qubits_map_reverse = {}         # Reverse qubit map

        self.qpugroup = None

        self.Num_Entanglement_swapping = 0          # Entanglement swapping count
        self.Num_RemoteGate = 0

        self.timeline = None                        # Scheduled timeline of result_circuit
        self.result_circuit_idle = None             # result_circuit + idle decoherence

    # Execute the distributed-circuit workflow.
    def Execution(self, config, qpugroup, comm_noise = False, estimate_time = True, idle_noise = False):
        self.qpugroup = qpugroup
        qpus = self.qpugroup.qpus

        self.split(config)
        self.valid_trans()
        self.check_swap_entanglement()
        print("Entanglement Number:", self.Num_Entanglement_swapping)
        self.rearrange_with_partition()
        self.rewrite_cross_group_cnots()
        self.physic_split()
        
        self.transpile_subcircuits(qpus)
        # for sub_circ in self.sub_circuit_trans:
        #     sub_circ.draw("mpl", scale = 0.5, fold = 100)
        #     plt.show()

        # dag = circuit_to_dag(self.sub_circuit_trans[2])
        # dag.draw(output='mpl')  
        # plt.show()

        result_qc = self.merge_trans_circuits(comm_noise)

        if estimate_time or idle_noise:
            self.schedule()
            print(f"Estimated execution time: {self.timeline.makespan * 1e6:.3f} us")

        # Waiting time -> T1/T2 decoherence on idle qubits
        if idle_noise:
            decoherence = idle_noise if isinstance(idle_noise, DecoherenceModel) else None
            result_qc = self.add_idle_noise(decoherence)
        
        return result_qc

    # Global qubit -> (QPU, physical qubit on that QPU)
    def _qubit_location(self):
        return {
            global_q: (self.qpus[sub_index], local_index)
            for global_q, (sub_index, local_index) in self.merged_qubits_map.items()
        }

    # Insert idle decoherence into the merged circuit based on the timeline
    def add_idle_noise(self, decoherence=None):
        """
        :param decoherence: DecoherenceModel (default: thermal relaxation with backend T1/T2)
        :return: result_circuit with idle-noise channels (also stored in self.result_circuit_idle)
        """
        if self.timeline is None:
            self.schedule()
        self.result_circuit_idle = add_idle_noise(
            self.result_circuit, self.timeline, self._qubit_location(), decoherence
        )
        return self.result_circuit_idle

    # Schedule the merged circuit with the timing interface
    def schedule(self, timing=None):
        """
        Build an ASAP timeline of self.result_circuit.

        :param timing: TimingProvider; defaults to self.qpugroup.timing
        :return: Timeline (makespan, events, per-qubit idle intervals)
        """
        if self.result_circuit is None:
            raise ValueError("No merged circuit found. Please run Execution first.")
        if timing is None:
            timing = self.qpugroup.timing

        self.timeline = DQCScheduler(timing).schedule(self.result_circuit, self._qubit_location())
        return self.timeline

    # Get the index of a qubit in the circuit
    def get_index(self, q):
        return int(self.find_bit(q).index)

    # Split the circuit based on the provided configuration
    def split(self, config):
        partition = []

        if all(isinstance(x, int) for x in config):
            # Generate index groups by group size.
            start = 0
            for s in config:
                indices = list(range(start, start + s))
                partition.append(indices)
                start += s
        elif all(isinstance(x, (list, tuple)) for x in config):
            # Use the specified index groups and sort each group.
            for sub in config:
                partition.append(sorted(sub))
        else:
            raise ValueError("Input must be a list of ints or a list of lists of ints")
        
        self.partition = partition

        for gid, group in enumerate(partition):
            self.Entanglement_swapping.append(0)
            for q in group:
                self.qubit_group[q] = gid
        return partition

    # Preprocessing to validate and decompose cross-group multi-qubit gates to single-qubit gates and CX gates
    # step[1]
    def valid_trans(self):
        """
        Walk through old_circ instructions and detect cross-group multi-qubit gates.
        Cross-group non-CX gates are decomposed before being appended to new_circ.
        Same-group gates and CX gates are appended directly.
        
        Args:
            new_circ: Target DQCCircuit.
            old_circ: Source DQCCircuit from step[0].
        """
        old_circ = self

         # === 1. Build a new circuit with quantum bits only. ===
        new_circ = DQCCircuit(old_circ.qubits)
        new_circ.qubit_group = old_circ.qubit_group
        new_circ.qubit_tele = old_circ.qubit_tele
        print("Qubit Group:", new_circ.qubit_group)

        # === 2. Preserve all classical registers. ===
        # Copy communication registers, if any.
        comm_cregs = [creg for creg in getattr(old_circ, "cregs", []) if "Tele" in creg.name]
        for comm_creg in comm_cregs:
            new_circ.add_register(ClassicalRegister(len(comm_creg), comm_creg.name))

        # Copy original registers, if any.
        orig_cregs = [creg for creg in getattr(old_circ, "cregs", []) if "Tele" not in creg.name]
        for orig_creg in orig_cregs:
            new_circ.add_register(ClassicalRegister(len(orig_creg), orig_creg.name))

        for instri in old_circ.data:
            instr = instri.operation   # Quantum gate or operation object
            qargs = instri.qubits      # Target quantum bits
            cargs = instri.clbits      # Target classical bits
            
            # Handle CX first.
            if instr.name in ["cx"]:
                new_circ.append(instr, qargs, cargs)
                continue

            # Append single-qubit gates directly.
            if len(qargs) <= 1:
                new_circ.append(instr, qargs, cargs)
                continue
            
            # Check whether a multi-qubit gate crosses groups.
            groups = [new_circ.qubit_group[old_circ.get_index(q)] for q in qargs]
            if len(set(groups)) > 1:
                # Decompose cross-group non-CX gates.
                ci = CircuitInstruction(instr, qargs, cargs)
                decomposed_instrs = self.decompose_and_get_data(ci)
                for di in decomposed_instrs:
                    mapped_qubits = [new_circ.qubits[old_circ.get_index(q)] for q in di.qubits]
                    mapped_clbits = [new_circ.clbits[old_circ.get_index(c)] for c in di.clbits]
                    new_circ.append(di.operation, mapped_qubits, mapped_clbits)
            else:
                # Append same-group multi-qubit gates directly.
                new_circ.append(instr, qargs, cargs)
        
        self.step.append(new_circ)
        return new_circ
    
    # step[2]
    def check_swap_entanglement(self):
        """Check cross-QPU entanglement and find a SWAP route when direct links are missing."""
        swap_routes = []
        qpu_map = self.qpugroup.map

        old_circ = self.step[1]

         # === 1. Build a new circuit with quantum bits only. ===
        new_circ = DQCCircuit(old_circ.qubits)

        for creg in getattr(old_circ, "cregs", []):
            new_circ.add_register(ClassicalRegister(len(creg), creg.name))

        # Visit each CX operation in the circuit.
        for instr in old_circ.data:
            if instr.operation.name == "cx":
                ctrl, tgt = instr.qubits
                g_ctrl = self.qubit_group[self.get_index(ctrl)]
                g_tgt = self.qubit_group[self.get_index(tgt)]

                # If control and target are on different QPUs.
                if g_ctrl != g_tgt:
                    # Check whether the QPUs are directly connected.
                    if not self.qpugroup.check_connection(g_ctrl, g_tgt):
                        # --- No direct link: find the shortest route. ---
                        path = self._find_shortest_path(qpu_map, g_ctrl, g_tgt)
                        if path:
                            swap_routes.append(path)
                            new_circ.append(S_CX(g_ctrl, g_tgt, path), instr.qubits, instr.clbits)
                            print(f"[Info] Found SWAP route {path} for CX({g_ctrl}, {g_tgt})")
                            self.Num_Entanglement_swapping += len(path) - 2
                        else:
                            print(f"[Warning] No route found between QPU {g_ctrl} and {g_tgt}")
                    else:
                        # Direct link: append the CX operation.
                        new_circ.append(instr.operation, instr.qubits, instr.clbits)
                else:
                    # Same group: append the CX operation.
                    new_circ.append(instr.operation, instr.qubits, instr.clbits)        
            else:
                new_circ.append(instr.operation, instr.qubits, instr.clbits)

        for path in swap_routes:
            # Mark intermediate nodes, excluding the source and destination.
            for node in path[1:-1]:
                self.Entanglement_swapping[node] = 1
                
        self.swap_routes = swap_routes
        self.step.append(new_circ)

    def _find_shortest_path(self, graph, start, end):
        """Find the shortest path in an adjacency-list graph with BFS."""
        from collections import deque

        visited = set()
        queue = deque([[start]])  # Each queue item is a path list.

        while queue:
            path = queue.popleft()
            node = path[-1]
            if node == end:
                return path  # Path found.

            if node not in visited:
                visited.add(node)
                for neighbor, _ in graph.get(node, []):  # Visit neighbors.
                    if neighbor not in visited:
                        queue.append(path + [neighbor])

        return None  # No path found.

    # step[3]
    def rearrange_with_partition(self):
        """
        Rearrange qubits according to partition, add one communication qubit per group,
        and allocate classical bits for inter-group communication.
        """
        if not self.partition:
            raise ValueError("Please set self.partition first.")

        partition = self.partition
        num_groups = len(partition)

        # Number of classical bits = g * (g - 1).
        comm_creg = ClassicalRegister(num_groups * (num_groups - 1), "Tele")

        # Number of qubits per group = original qubits + 1 communication qubit.

        # Count regular qubits in the partition.
        total_qubits = sum(len(group) for group in partition)
        # Add one communication qubit per group.
        total_qubits += len(partition)
        # Add one extra communication qubit for each QPU that needs entanglement swapping.
        total_qubits += self.Entanglement_swapping.count(1)

        new_qreg = QuantumRegister(total_qubits, "q")

        new_circ = DQCCircuit(new_qreg, comm_creg)

        old_circ = self.step[2]
        # === 4. Preserve original classical-register metadata. ===
        # Preserve original classical registers when they exist.
        if hasattr( old_circ, "clbits") and old_circ.clbits:
            # Try to reuse the original register name from self.cregs.
            if hasattr( old_circ, "cregs") and old_circ.cregs:
                # Use the name of the first ClassicalRegister.
                orig_name =  old_circ.cregs[0].name
            else:
                orig_name = "c"  # Default to "c" when no name is recorded.

            # Create a new ClassicalRegister with the original name.
            orig_creg = ClassicalRegister(len(old_circ.clbits), orig_name)
            new_circ.add_register(orig_creg)

            # Keep a reference for later access.
            new_circ.orig_creg = orig_creg
        else:
            new_circ.orig_creg = None

        # ===== Build the old-qubit to new-qubit mapping. =====
        old2new = {}
        group_comm_qubits = []
        new_index = 0

        for indices in partition:
            # Map quantum-register entries.
            # Qubits inside the group.
            for qi in indices:
                old2new[qi] = new_qreg[new_index]   
                new_index += 1
            # comm qubit
            comm_q = new_qreg[new_index]
            group_comm_qubits.append(comm_q)
            new_index += 1
            if self.Entanglement_swapping[self.qubit_group[indices[0]]] == 1:
                # Add one extra communication qubit for QPUs that need entanglement swapping.
                extra_comm_q = new_qreg[new_index]
                group_comm_qubits.append(extra_comm_q)
                new_index += 1

        # ===== Visit the original circuit and remap operations to the new circuit. =====
        for instr in old_circ.data:
            qubit_indices = [old_circ.get_index(q) for q in instr.qubits]

            if all(qi in old2new for qi in qubit_indices):
                new_qargs = [old2new[qi] for qi in qubit_indices]
                new_circ.append(instr.operation, new_qargs, instr.clbits)

        # ==== Populate new_circ.qubit_group. =====
        new_circ.qubit_group = []
        for gid, group in enumerate(partition):
            base_count = len(group) + 1  # Regular + communication qubits
            if self.Entanglement_swapping[gid] == 1:
                base_count += 1  # Add one extra SWAP qubit.
            new_circ.qubit_group.extend([gid] * base_count)

        # ===== Populate new_circ.qubit_tele. =====
        qubit_tele = [-1] * len(new_qreg) 

        idx = 0
        for gid, group in enumerate(partition):
            # Regular and communication qubits for each partition.
            comm_q = group_comm_qubits[idx]
            comm_idx = new_circ.get_index(comm_q)

            # Regular qubits point to the communication qubit for their group.
            for qi in group:
                qubit_tele[new_circ.get_index(old2new[qi])] = comm_idx

            # Communication qubits point to -1.
            qubit_tele[comm_idx] = -1
            idx += 1

            # Handle the extra communication qubit for groups that need entanglement swapping.
            if self.Entanglement_swapping[gid] == 1:
                swap_comm_q = group_comm_qubits[idx]
                swap_comm_idx = new_circ.get_index(swap_comm_q)
                qubit_tele[swap_comm_idx] = -1  # The swapping communication qubit also points to -1.
                idx += 1

        # Assign the communication-qubit map.
        new_circ.qubit_tele = qubit_tele

        self.step.append(new_circ)
        return new_circ

    # Rewrite cross-group CNOTs into RemoteGate, Measurement, If_X, If_Z
    # step[4]
    def rewrite_cross_group_cnots(self):
        
        old_circ = self.step[3]

        # === 1. Build a new circuit with quantum bits only. ===
        new_circ = DQCCircuit(old_circ.qubits)
        new_circ.qubit_group = old_circ.qubit_group
        new_circ.qubit_tele = old_circ.qubit_tele
        group_tele = {gid: self.step[3].qubit_tele[self.step[3].qubit_group.index(gid)] for gid in set(self.step[3].qubit_group)}

        # === 2. Preserve all classical registers. ===
        # Copy communication registers, if any.
        comm_cregs = [creg for creg in getattr(old_circ, "cregs", []) if "Tele" in creg.name]
        for comm_creg in comm_cregs:
            new_circ.add_register(ClassicalRegister(len(comm_creg), comm_creg.name))

        # Copy original registers, if any.
        orig_cregs = [creg for creg in getattr(old_circ, "cregs", []) if "Tele" not in creg.name]
        for orig_creg in orig_cregs:
            new_circ.add_register(ClassicalRegister(len(orig_creg), orig_creg.name))


        idx_counter = 0  # Global index counter

        # === 4. Rewrite the circuit. ===
        for inst_obj in old_circ.data:
            instr = inst_obj.operation
            qargs = inst_obj.qubits
            cargs = inst_obj.clbits
            if instr.name == "cx": 
                ctrl, tgt = qargs
                g_ctrl = new_circ.qubit_group[old_circ.get_index(ctrl)]
                g_tgt = new_circ.qubit_group[old_circ.get_index(tgt)]

                if g_ctrl != g_tgt:
                    # ====== Rewrite cross-group CNOT. ======
                    ctrl_index = old_circ.get_index(ctrl)
                    tgt_index = old_circ.get_index(tgt)
                    ctrl_comm_index = old_circ.qubit_tele[ctrl_index]
                    tgt_comm_index = old_circ.qubit_tele[tgt_index]

                    ctrl_comm = new_circ.qubits[ctrl_comm_index]
                    tgt_comm = new_circ.qubits[tgt_comm_index]
                    ctrl_q = new_circ.qubits[ctrl_index]
                    tgt_q = new_circ.qubits[tgt_index]

                    # Reset 
                    new_circ.append(Reset(), [ctrl_comm])
                    new_circ.append(Reset(), [tgt_comm])

                    # RemoteGate
                    new_circ.append(RemoteGate(idx_counter, g_tgt), [ctrl_comm])
                    new_circ.append(RemoteGate(idx_counter, g_ctrl), [tgt_comm])
                    idx_counter += 1

                    # CNOT: ctrl->ctrl_comm, tgt_comm->tgt
                    new_circ.append(CXGate(), [ctrl_q, ctrl_comm])
                    new_circ.append(CXGate(), [tgt_comm, tgt_q])

                    # H on tgt_comm
                    new_circ.append(HGate(), [tgt_comm])

                    # Measurement + If_Z
                    # Measurement + If_X
                    mz_inst = MZ(index=idx_counter, target=g_tgt)
                    mx_inst = MX(index=idx_counter, target=g_ctrl)
                    new_circ.append(mz_inst, [ctrl_q, ctrl_comm])
                    new_circ.append(mx_inst, [tgt_comm, tgt_q])
                    idx_counter += 1

                    self.Num_RemoteGate += 1
                else:
                    # ====== Keep same-group CNOT unchanged. ======
                    new_circ.append(instr, qargs, cargs)
            elif instr.name == "measure":
                # ====== Replace measurement with AnsM. ======
                qarg = qargs[0]
                carg = cargs[0] if cargs else None

                # Get the classical bit index in its register.
                if carg is not None:
                    mea_index = old_circ.find_bit(carg).index
                else:
                    mea_index = -1  # No associated classical bit.

                # Create the AnsM placeholder instruction.
                ansm_gate = AnsM(mea_index)

                # Add it to the new circuit.
                new_circ.append(ansm_gate, [qarg])
            elif instr.name == "S_CX":
                path = instr.path
                ctrl, tgt = qargs
                g_ctrl = new_circ.qubit_group[old_circ.get_index(ctrl)]
                g_tgt = new_circ.qubit_group[old_circ.get_index(tgt)]

                ctrl_index = old_circ.get_index(ctrl)
                tgt_index = old_circ.get_index(tgt)
                ctrl_comm_index = old_circ.qubit_tele[ctrl_index]
                tgt_comm_index = old_circ.qubit_tele[tgt_index]

                ctrl_q = new_circ.qubits[ctrl_index]
                tgt_q = new_circ.qubits[tgt_index]
                ctrl_comm = new_circ.qubits[ctrl_comm_index]
                tgt_comm = new_circ.qubits[tgt_comm_index]

                new_circ.append(Reset(), [ctrl_comm_index])

                for i in range(1, len(path) - 1):
                    mid_g = path[i]
                    tgt_g = path[i + 1]

                    mid_comm_index_1 = group_tele[mid_g]
                    mid_comm_index_2 = mid_comm_index_1 + 1
                    tgt_comm_index = group_tele[tgt_g]

                    mid_comm_1 = new_circ.qubits[mid_comm_index_1]
                    mid_comm_2 = new_circ.qubits[mid_comm_index_2]
                    tgt_comm = new_circ.qubits[tgt_comm_index]

                    if i == 1:
                        # Reset 
                        new_circ.append(Reset(), [mid_comm_1])
                        new_circ.append(Reset(), [mid_comm_2])
                        new_circ.append(Reset(), [tgt_comm])

                        # RemoteGate
                        new_circ.append(RemoteGate(idx_counter, mid_g), [ctrl_comm])
                        new_circ.append(RemoteGate(idx_counter, g_ctrl), [mid_comm_1])
                        idx_counter += 1

                        new_circ.append(RemoteGate(idx_counter, tgt_g), [mid_comm_2])
                        new_circ.append(RemoteGate(idx_counter, mid_g), [tgt_comm])
                        idx_counter += 1
                        
                        new_circ.append(CXGate(), [mid_comm_1, mid_comm_2])
                        new_circ.append(HGate(), [mid_comm_1])

                        new_circ.append(IF_Z(index=idx_counter, target=mid_g), [ctrl_comm_index])
                        new_circ.append(MS(index=idx_counter, target=tgt_g), [mid_comm_1, mid_comm_2])
                        new_circ.append(IF_X(index=idx_counter, target=g_ctrl), [tgt_comm_index])
                        idx_counter += 1
                    else:
                        new_circ.append(Reset(), [mid_comm_2])
                        new_circ.append(Reset(), [tgt_comm])

                        # RemoteGate
                        new_circ.append(RemoteGate(idx_counter, tgt_g), [mid_comm_2])
                        new_circ.append(RemoteGate(idx_counter, mid_g), [tgt_comm])
                        idx_counter += 1
                        
                        new_circ.append(CXGate(), [mid_comm_1, mid_comm_2])
                        new_circ.append(HGate(), [mid_comm_1])

                        new_circ.append(IF_Z(index=idx_counter, target=mid_g), [ctrl_comm_index])
                        new_circ.append(MS(index=idx_counter, target=tgt_g), [mid_comm_1, mid_comm_2])
                        new_circ.append(IF_X(index=idx_counter, target=g_ctrl), [tgt_comm_index])
                        idx_counter += 1             

                
                # CNOT: ctrl->ctrl_comm, tgt_comm->tgt
                new_circ.append(CXGate(), [ctrl_q, ctrl_comm])
                new_circ.append(CXGate(), [tgt_comm, tgt_q])

                # H on tgt_comm
                new_circ.append(HGate(), [tgt_comm])

                # Measurement + If_Z
                # Measurement + If_X
                mz_inst = MZ(index=idx_counter, target=g_tgt)
                mx_inst = MX(index=idx_counter, target=g_ctrl)
                new_circ.append(mz_inst, [ctrl_q, ctrl_comm])
                new_circ.append(mx_inst, [tgt_comm, tgt_q])
                idx_counter += 1   
            else:
                # Keep non-CNOT and non-measurement instructions unchanged.
                new_circ.append(instr, qargs, cargs)

        self.step.append(new_circ)
        return new_circ

    # Physically split the circuit into sub-circuits based on partition
    def physic_split(self):
        config = [len(group) + 1 for group in self.partition]
        for gid, flag in enumerate(self.Entanglement_swapping):
            if flag == 1:
                config[gid] += 1

        temp_circuit = self.step[4]
        sub_circuits = []

        if all(isinstance(x, int) for x in config):
            # Split by group sizes in order, for example [2, 2, 3].
            start = 0
            for s in config:
                sub = DQCCircuit(s)
                sub.qubit_group = temp_circuit.qubit_group[start:start+s]

                # Add classical bits to the sub-circuit.
                num_clbits = (len(config) - 1) * 2
                if num_clbits > 0:
                    creg = ClassicalRegister(num_clbits, "c")
                    sub.add_register(creg)

                # Copy operations from the corresponding qubits in the source circuit.
                for instr in temp_circuit.data:
                    qubit_indices = [temp_circuit.get_index(q) for q in instr.qubits]
                    # Keep only qubits that belong to this sub-circuit.
                    indices_in_sub = [i for i, qi in enumerate(qubit_indices) if start <= qi < start+s]
                    if indices_in_sub:
                        new_qargs = [sub.qubits[qi - start] for i, qi in enumerate(qubit_indices) if i in indices_in_sub]
                        new_cargs = []
                        # print(f"Appending instruction {instr.operation.name} to sub-circuit with qubits {new_qargs}")
                        sub.append(instr.operation, new_qargs, new_cargs)
                        
                sub_circuits.append(sub)
                start += s

        elif all(isinstance(x, (list, tuple)) for x in config):
            # Split by explicit index groups, for example [[1, 3], [0, 2, 4]].
            for indices in config:
                sub = DQCCircuit(len(indices))
                sub.qubit_type = [self.qubit_type[i] for i in indices]

                for instr in self.data:
                    qubit_indices = [self.get_index(q) for q in instr.qubits]
                    # Keep only qubits included in indices.
                    indices_in_sub = [i for i, qi in enumerate(qubit_indices) if qi in indices]
                    if indices_in_sub:
                        new_qargs = [sub.qubits[indices.index(qi)] for i, qi in enumerate(qubit_indices) if i in indices_in_sub]
                        new_cargs = []
                        sub.append(instr.operation, new_qargs, new_cargs)

                sub_circuits.append(sub)

        else:
            raise ValueError("Config must be a list of ints or a list of lists of ints")

        self.sub_circuit = sub_circuits
        return sub_circuits

    # Protect custom instructions with barriers
    def protect_custom_instructions(self, subcirc):
        """
        Add barrier protection around RemoteGate, Measurement, If_X, If_Z, and AnsM.
        """
        new_circ = DQCCircuit(*subcirc.qregs, *subcirc.cregs)
        
        for inst_obj in subcirc.data:
            inst = inst_obj.operation
            qargs = inst_obj.qubits
            cargs = inst_obj.clbits

            if isinstance(inst, (MX, MZ, AnsM, IF_Z, IF_X, MS, S_CX)):               # Add barriers on the same qubits.
                new_circ.barrier(*qargs)
                new_circ.append(inst, qargs, cargs)
                new_circ.barrier(*qargs)
                # print(f"[Info] Protecting instruction: {inst.name} on qubits {[q for q in qargs]}")
            elif isinstance(inst, (RemoteGate)):
                new_circ.barrier()
                new_circ.append(inst, qargs, cargs)
                new_circ.barrier()
            else:
                new_circ.append(inst, qargs, cargs)

        return new_circ

    # Restore custom instructions by removing barriers
    def restore_custom_instructions(self, subcirc):
        """
        Remove barrier protection around RemoteGate, Measurement, If_X, If_Z, and AnsM.
        """
        new_circ = DQCCircuit(*subcirc.qregs, *subcirc.cregs)
        
        for inst_obj in subcirc.data:
            inst = inst_obj.operation
            qargs = inst_obj.qubits
            cargs = inst_obj.clbits

            if inst.name != "barrier":
                new_circ.append(inst, qargs, cargs)
            # if isinstance(inst, (RemoteGate, MX, MZ, AnsM)):  
            #     print(f"[Info] Restoring instruction: {inst.name} on qubits {[q for q in qargs]}")
        return new_circ
    
    def transpile_subcircuits(self, qpus, layout_out=None):
        """
        Transpile each sub-circuit using the corresponding QPU backend target,
        and optional layout mapping per subcircuit.

        :param qpus: list of QPU instances (each having .backend and .target)
        :param layout_out: list of layouts; each layout is a list of physical qubit indices
                        corresponding to the sub-circuit's logical qubits
                        e.g. [[0,1,2,3], [5,6,7,8]]
        """
        # ---- Sort by qpu_id. ----
        self.qpus = sorted(qpus, key=lambda x: x.qpu_id)

        if not self.sub_circuit:
            raise ValueError("No sub-circuits found. Please populate self.sub_circuit first.")
        if len(qpus) != len(self.sub_circuit):
            raise ValueError(
                f"Number of QPUs ({len(qpus)}) does not match number of sub-circuits ({len(self.sub_circuit)})."
            )

        # ---- Validate layout_out. ----
        if layout_out is not None:
            if len(layout_out) != len(self.sub_circuit):
                raise ValueError("layout_out length must match number of sub-circuits.")

        # ---- Check whether each sub-circuit fits its backend. ----
        for idx, (sub_circ, qpu) in enumerate(zip(self.sub_circuit, qpus)):
            backend = qpu.backend
            size = sub_circ.num_qubits
            qpu_id = getattr(qpu, "qpu_id", idx)
            backend_name = getattr(backend, "name", "unknown")

            if size > backend.num_qubits:
                raise ValueError(
                    f"QPU {qpu_id}: requested size={size} exceeds "
                    f"the maximum number of qubits ({backend.num_qubits}) "
                    f"supported by backend {backend_name}."
                )

        self.sub_circuit_trans = []

        # ---- Transpile each sub-circuit. ----
        for idx, (sub, qpu) in enumerate(zip(self.sub_circuit, qpus)):
            try:
                # Step 1: Protect custom instructions.
                sub = self.protect_custom_instructions(sub)

                layout = None
                if layout_out is not None and idx < len(layout_out):
                    layout = layout_out[idx]

                # Step 3: Call transpile.
                if layout is not None:
                    sub = transpile(
                        sub,
                        backend=qpu.backend,
                        initial_layout=layout,
                        optimization_level=3,
                    )
                else:
                    sub = transpile(
                        sub,
                        backend=qpu.backend,
                        optimization_level=3,
                    )

                # Step 4: Restore custom instructions.
                sub = self.restore_custom_instructions(sub)

                self.sub_circuit_trans.append(sub)
                print(f"[Info] Sub-circuit {idx} transpiled successfully on {qpu.backend.name}.")

            except Exception as e:
                print(f"[Error] Failed to transpile sub-circuit {idx} on {qpu.backend.name}: {e}")
                self.sub_circuit_trans.append(None)

        return self.sub_circuit_trans
    
    # Merge the transpiled sub-circuits into a complete circuit
    def merge_trans_circuits(self, comm_noise = False):
        """
        Combine sub-circuits from self.sub_circuit_trans into a complete circuit.
        Supports cross-circuit paired operations (R, M+IF_X/IF_Z) and uses a stack
        to return to the caller circuit.
        Bidirectional pairing works no matter whether M or IF_X/IF_Z is visited first.
        """
        # === Build qubit and cbit maps. ===
        qubits_map = {}
        merged_qubits_map = {}
        cbits_map = {}
        global_q_index = 0
        global_c_index = 0
        for i, sub in enumerate(self.sub_circuit_trans):
            if sub is None:
                continue
            local_indices = {sub.find_bit(q).index for instr in sub.data for q in instr.qubits}
            for local_index in sorted(local_indices):
                qubits_map[(i, local_index)] = global_q_index
                merged_qubits_map[global_q_index] = (i, local_index)
                global_q_index += 1

        # Record index for noise model
        self.merged_qubits_map = merged_qubits_map
        self.merged_qubits_map_reverse = qubits_map

        num_sub = len(self.sub_circuit_trans)
        for i in range(num_sub):
            for j in range(num_sub):
                if i != j:
                    cbits_map[(i, j)] = global_c_index
                    global_c_index += 1

        # === Initialize state. ===
        # Create a new circuit with quantum bits only.
        new_circ = DQCCircuit(len(qubits_map))
        new_circ.qubit_group = self.step[1].qubit_group

        # === 1. Add the communication register. ===
        tele_creg = ClassicalRegister(len(cbits_map), "Tele")
        new_circ.add_register(tele_creg)

        # === 2. Preserve original classical registers. ===
        if hasattr(self, "clbits") and self.clbits:
            if hasattr(self, "cregs") and self.cregs:
                orig_name = self.cregs[0].name
            else:
                orig_name = "c"
            orig_creg = ClassicalRegister(len(self.clbits), orig_name)
            new_circ.add_register(orig_creg)

        indices = [0] * len(self.sub_circuit_trans)
        paired_op = {}      # {index: (sub_index, instr)}
        paired_op2 = {}      
        paired_done = set() # Completed pairing indices.
        call_stack = []
        now = 0
        step = 0

        # Helper for conditional operations.
        def apply_conditional_gate(gate_list, qubit, clbit):
            with new_circ.if_test((clbit, 1)):
                for instr in gate_list:
                    new_circ.append(instr.operation, [qubit], instr.clbits)

        while True:
            done = all(sub is None or indices[i] >= len(sub.data) for i, sub in enumerate(self.sub_circuit_trans))
            if done:
                break

            sub = self.sub_circuit_trans[now]
            if sub is None or indices[now] >= len(sub.data):
                if call_stack:
                    now = call_stack.pop()
                    continue
                else:
                    now = (now + 1) % len(self.sub_circuit_trans)
                    continue

            instr = sub.data[indices[now]]
            inst = instr.operation
            op_name = inst.name.upper()
            local_indices = [sub.find_bit(q).index for q in instr.qubits]
            global_qs = [qubits_map[(now, idx)] for idx in local_indices]
            target = getattr(inst, "target", None)
            idx = getattr(inst, "index", None)
            mea = getattr(inst,"mea", None)

            # print(f"\n[STEP] now={now}, target={target},idx={idx}, inst={inst.name}, indices={indices}")

            # === Pair R gates to create a Bell state. ===
            if op_name == "R" and idx not in paired_done:
                target_sub = self.sub_circuit_trans[target] if target is not None else None

                if idx not in paired_op:
                    paired_op[idx] = now
                    if target_sub:
                        call_stack.append(now)
                        now = target
                        continue
                else:
                    first_now = paired_op.pop(idx)
                    first_sub = self.sub_circuit_trans[first_now]
                    first_instr = first_sub.data[indices[first_now]]
                    first_qs = [qubits_map[(first_now, first_sub.find_bit(q).index)] for q in first_instr.qubits]
                    target_qs = global_qs
                    # print(first_now,now,first_qs,target_qs)
                    new_circ.initialize([1/np.sqrt(2),0,0,1/np.sqrt(2)], first_qs + target_qs)
                    if comm_noise:
                        noise_instr = self.qpugroup.get_noise_instruction(first_now, now)
                        # Insert noise mapped to the new circuit qubits.
                        if isinstance(target_qs, list):
                            # Apply noise to all qubits in the list.
                            noise_qargs = [new_circ.qubits[i] for i in target_qs]
                        else:
                            # Apply noise to one qubit.
                            noise_qargs = [new_circ.qubits[target_qs]]

                        if noise_instr is not None:
                            new_circ.append(noise_instr, noise_qargs)

                    paired_done.add(idx)
                    indices[now] += 1
                    indices[first_now] += 1
                    now = call_stack.pop() if call_stack else now
                    continue

            # === Bidirectional M / IF_X / IF_Z pairing. ===
            elif op_name in ("MX", "MZ") and idx not in paired_done:
                if idx not in paired_op:
                    paired_op[idx] = now
                    if target_sub:
                        call_stack.append(now)
                        now = target
                    continue

                # Pair already found.
                first_now = paired_op.pop(idx)
                first_sub = self.sub_circuit_trans[first_now]
                first_instr = first_sub.data[indices[first_now]]

                first_qs = [qubits_map[(first_now, first_sub.find_bit(q).index)] 
                            for q in first_instr.qubits]
                target_qs = global_qs

                # Classical-bit mapping.
                cbit_idx1 = cbits_map[(now, first_now)]
                cbit_idx2 = cbits_map[(first_now, now)]
                clbit_obj1 = new_circ.clbits[cbit_idx1]
                clbit_obj2 = new_circ.clbits[cbit_idx2]


                if first_instr.name == "MX":
                    new_circ.measure(first_qs[0], clbit_obj1)
                    apply_conditional_gate(self.qpus[now].compile_z_gate(), target_qs[0], clbit_obj1)
                    new_circ.measure(target_qs[1], clbit_obj2)
                    apply_conditional_gate(self.qpus[first_now].compile_x_gate(), first_qs[1], clbit_obj2)

                else:
                    new_circ.measure(first_qs[1], clbit_obj1)
                    apply_conditional_gate(self.qpus[now].compile_x_gate(), target_qs[1], clbit_obj1)
                    new_circ.measure(target_qs[0], clbit_obj2)
                    apply_conditional_gate(self.qpus[first_now].compile_z_gate(), first_qs[0], clbit_obj2)
                    

                # Update state.
                paired_done.add(idx)
                indices[now] += 1
                indices[first_now] += 1
                now = call_stack.pop() if call_stack else now
                continue
            elif op_name == "ANS_M":
                # global_qs[0] is the qubit index; mea stores the classical-bit index.
                q_idx = global_qs[0]
                c_idx = mea  # mea is already a global classical-bit index.

                # Find the corresponding Clbit object in new_circ.
                clbit_obj = None
                temp_idx = c_idx
                for creg in new_circ.cregs:
                    if temp_idx < len(creg):
                        clbit_obj = creg[temp_idx]
                        break
                    else:
                        temp_idx -= len(creg)

                if clbit_obj is None:
                    raise ValueError(f"Cannot find the classical bit index {c_idx} corresponding to a Clbit")

                # Add the measurement operation.
                new_circ.measure(new_circ.qubits[q_idx], clbit_obj)
                indices[now] += 1
                continue
            elif op_name in ("IF_X", "IF_Z", "MS") and idx not in paired_done:
                target_sub = self.sub_circuit_trans[target] if target is not None else None
                if idx not in paired_op:
                    paired_op[idx] = now
                    if target_sub:
                        call_stack.append(now)
                        now = target
                    continue
                if idx not in paired_op2:
                    paired_op2[idx] = now
                    if target_sub:
                        call_stack.append(now)
                        now = target
                    continue

                # Pair already found.
                first_now = paired_op.pop(idx)
                first_sub = self.sub_circuit_trans[first_now]
                first_instr = first_sub.data[indices[first_now]]

                second_now = paired_op2.pop(idx)
                second_sub = self.sub_circuit_trans[second_now]
                second_instr = second_sub.data[indices[second_now]]

                first_qs = [qubits_map[(first_now, first_sub.find_bit(q).index)] 
                            for q in first_instr.qubits]
                
                second_qs = [qubits_map[(second_now, second_sub.find_bit(q).index)] 
                            for q in second_instr.qubits]
                
                # instr global_qs
                third_qs = [qubits_map[(now, self.sub_circuit_trans[now].find_bit(q).index)] 
                            for q in instr.qubits]

                # Classical-bit mapping.
                cbit_idx1 = cbits_map[(first_now, second_now)]  
                cbit_idx2 = cbits_map[(second_now, first_now)]  
                cbit_idx3 = cbits_map[(second_now, now)]        
                cbit_idx4 = cbits_map[(now, second_now)]                
                cbit_idx5 = cbits_map[(now, first_now)]         
                cbit_idx6 = cbits_map[(first_now, now)]         

                clbit_obj1 = new_circ.clbits[cbit_idx1]
                clbit_obj2 = new_circ.clbits[cbit_idx2] 
                clbit_obj3 = new_circ.clbits[cbit_idx3]
                clbit_obj4 = new_circ.clbits[cbit_idx4] 
                clbit_obj5 = new_circ.clbits[cbit_idx5]
                clbit_obj6 = new_circ.clbits[cbit_idx6] 
                # print("Entanglement Swapping", first_instr.name,second_instr.name, instr.name)
                if first_instr.name == "IF_Z":
                    new_circ.measure(second_qs[0], clbit_obj2)
                    apply_conditional_gate(self.qpus[first_now].compile_z_gate(), first_qs, clbit_obj2)
                    new_circ.measure(second_qs[1], clbit_obj3)
                    apply_conditional_gate(self.qpus[now].compile_x_gate(), third_qs, clbit_obj3)
                elif first_instr.name == "MS":
                    new_circ.measure(first_qs[0], clbit_obj6)
                    apply_conditional_gate(self.qpus[now].compile_z_gate(), third_qs, clbit_obj6)
                    new_circ.measure(first_qs[1], clbit_obj1)
                    apply_conditional_gate(self.qpus[second_now].compile_x_gate(), second_qs, clbit_obj1)
                elif first_instr.name == "IF_X":
                    new_circ.measure(third_qs[0], clbit_obj4)
                    apply_conditional_gate(self.qpus[second_now].compile_z_gate(), second_qs, clbit_obj4)
                    new_circ.measure(third_qs[1], clbit_obj5)
                    apply_conditional_gate(self.qpus[first_now].compile_x_gate(), first_qs, clbit_obj5)
                # Update state.
                paired_done.add(idx)
                indices[now] += 1
                indices[first_now] += 1
                indices[second_now] += 1 
                now = call_stack.pop() if call_stack else now
                now = call_stack.pop() if call_stack else now
                continue
            # === Regular gate. ===
            else:
                new_circ.append(inst, global_qs, [])
                indices[now] += 1
                continue

            step += 1
            if step > 50000:
                raise RuntimeError(f"[Error] Possible deadlock. indices={indices}, now={now}, stack={call_stack}")

        self.result_circuit = new_circ
        self.result_circuit.qubit_group = self.step[4].qubit_group
        return new_circ
    
    def decompose_and_get_data(self, instr: CircuitInstruction, basis_gates=None):
        """
        Recursively decompose one instruction into the basis-gate set.
        
        Args:
            instr: CircuitInstruction to decompose.
            basis_gates: Allowed basis-gate names. Defaults to {'cx'} plus single-qubit gates.
        
        Returns:
            A list of CircuitInstruction objects preserving original qubit references.
        """
        if basis_gates is None:
            # Define the basis-gate set.
            basis_gates = {'cx', 'u3', 'u2', 'u1', 'id', 'x', 'y', 'z', 
                        'h', 's', 't', 'rx', 'ry', 'rz', 'sx', 'p'}
        
        # Return directly for basis gates or operations without a definition.
        if (not hasattr(instr.operation, "definition") or 
            instr.operation.name in basis_gates):
            return [instr]
        
        # Create a temporary circuit for decomposition.
        n_qubits = len(instr.qubits)
        qc_temp = QuantumCircuit(n_qubits)
        qc_temp.append(instr.operation, list(range(n_qubits)))
        
        # Map temporary qubits back to original qubits.
        # qc_temp.qubits[i] corresponds to instr.qubits[i].
        temp_to_orig = {qc_temp.qubits[i]: instr.qubits[i] for i in range(n_qubits)}
        
        # Decompose once.
        decomposed = qc_temp.decompose()
        
        result = []
        for sub_instr in decomposed.data:
            # Map temporary qubit objects back to original qubit objects.
            mapped_qubits = [temp_to_orig[q] for q in sub_instr.qubits]
            
            # Create a new instruction.
            new_instr = CircuitInstruction(
                sub_instr.operation, 
                mapped_qubits, 
                sub_instr.clbits
            )
            
            # Recursively decompose again if needed.
            result.extend(self.decompose_and_get_data(new_instr, basis_gates))
        
        return result

    def reduce_noise_model(self, subset_qubits, noise_model=None, coupling_map=None):
        """
        Reduce a NoiseModel so it keeps only noise on subset_qubits while preserving
        basis_gates, description, and the optional coupling map.
        
        Args:
            subset_qubits: list[int]
                Qubits whose noise should be retained.
            noise_model: Optional NoiseModel.
                Uses self.backend_noise_model when not provided.
            coupling_map: Optional list[tuple].
                If provided, the coupling map is reduced as well.

        Returns:
            NoiseModel
        """

        if noise_model is None:
            if getattr(self, "backend_noise_model", None) is None:
                raise ValueError("No noise model provided or set in self.backend_noise_model.")
            noise_model = self.backend_noise_model

        # Prefer Qiskit's reduce() method when available.
        if hasattr(noise_model, "reduce"):
            return noise_model.reduce(subset_qubits)

        sub_model = NoiseModel()

        # --- 1. Preserve quantum-gate noise. ---
        for instr_name, qerrors in noise_model._local_quantum_errors.items():
            for qubits, error in qerrors.items():
                if all(q in subset_qubits for q in qubits):
                    sub_model.add_quantum_error(error, instr_name, qubits)

        # --- 2. Preserve readout noise (measure). ---
        for qubit, error in noise_model._local_readout_errors.items():
            if qubit[0] in subset_qubits:  # qubit is a tuple, for example (0,).
                sub_model.add_readout_error(error, [qubit[0]])  

        # --- 3. Preserve basis_gates and description. ---
        if hasattr(noise_model, "basis_gates"):
            sub_model._basis_gates = list(noise_model.basis_gates)
        if hasattr(noise_model, "description"):
            sub_model._description = noise_model.description

        # --- 4. Optionally reduce the coupling map. ---
        target_coupling_map = coupling_map if coupling_map is not None else getattr(self, "coupling_map", None)
        if target_coupling_map is not None:
            sub_model._coupling_map = [edge for edge in target_coupling_map if all(q in subset_qubits for q in edge)]
        else:
            sub_model._coupling_map = None

        return sub_model

    # Get a combined noise model for the distributed circuit based on QPU backends
    def get_noise_model(self):
        """
        Build a combined noise model by reducing and merging noise models from
        multiple QPUs, then mapping them to global contiguous qubits.
        """
        # === Step 1: Validate inputs. ===
        qpus = self.qpugroup.qpus
        qpus = sorted(qpus, key=lambda x: x.qpu_id)
        if not self.sub_circuit:
            raise ValueError("No sub-circuits found.")
        if len(qpus) != len(self.sub_circuit):
            raise ValueError(
                f"Number of QPUs ({len(qpus)}) does not match sub-circuits ({len(self.sub_circuit)})."
            )

        # === Step 2: Extract local qubits for each sub-circuit. ===
        sub_qubit_maps = []
        for idx, _ in enumerate(self.sub_circuit):
            local_qubits = [
                local_index
                for global_q, (sub_index, local_index) in self.merged_qubits_map.items()
                if sub_index == idx
            ]
            sub_qubit_maps.append(local_qubits)

        # === Step 3: Build global qubit mapping (sub_idx, local_qubit) -> global_qubit. ===
        qubit_mapping = self.merged_qubits_map_reverse

        # === Step 4: Initialize the global NoiseModel. ===
        combined_noise_model = NoiseModel()
        combined_basis_gates = set()

        # === Step 5: Reduce and merge noise for each QPU. ===
        for idx, qpu in enumerate(qpus):
            backend_noise = NoiseModel.from_backend(qpu.backend)
            local_qubits = sub_qubit_maps[idx]

            # --- 5a. Reduce local noise. ---
            reduced_noise = self.reduce_noise_model(
                subset_qubits=local_qubits,
                noise_model=backend_noise,
                coupling_map=qpu.backend.coupling_map
            )

            # --- 5b. Merge quantum-gate noise. ---
            for instr_name, qerrors in reduced_noise._local_quantum_errors.items():
                for qubits_tuple, error in qerrors.items():
                    # Unpack qubits.
                    qubit_indices = [q if isinstance(q, int) else q[0] for q in qubits_tuple]
                    global_qubits = tuple(qubit_mapping[(idx, q)] for q in qubit_indices)
                    combined_noise_model.add_quantum_error(error, instr_name, global_qubits)
                    # print(f"Added quantum error: {instr_name}, local {qubit_indices} -> global {global_qubits}")

            # --- 5c. Merge readout noise. ---
            for qubit, error in reduced_noise._local_readout_errors.items():
                q_local = qubit[0] if isinstance(qubit, tuple) else qubit
                if (idx, q_local) not in qubit_mapping:
                    print(f"[Warning] Qubit mapping missing for {(idx, q_local)}")
                    continue
                global_qubit = qubit_mapping[(idx, q_local)]
                combined_noise_model.add_readout_error(error, [global_qubit])
                # print(f"Added readout error: local {q_local} -> global {global_qubit}")

            # --- 5d. Collect basis_gates. ---
            if hasattr(reduced_noise, "basis_gates"):
                combined_basis_gates.update(reduced_noise.basis_gates)

        # === Step 6: Store global basis gates and description. ===
        combined_noise_model._basis_gates = list(combined_basis_gates)
        combined_noise_model._description = "Combined noise model from multiple QPUs"

        return combined_noise_model
    
def detect_swap_pattern(circuit):
    swap_count = 0
    instructions = list(circuit.data)
    
    i = 0
    while i < len(instructions) - 2:
        inst1 = instructions[i]
        inst2 = instructions[i + 1]
        inst3 = instructions[i + 2]
        
        # Check whether all three instructions are CX gates.
        if (inst1.operation.name == 'cx' and 
            inst2.operation.name == 'cx' and 
            inst3.operation.name == 'cx'):
            
            # Get qubit indices.
            q1_0 = circuit.find_bit(inst1.qubits[0]).index
            q1_1 = circuit.find_bit(inst1.qubits[1]).index
            
            q2_0 = circuit.find_bit(inst2.qubits[0]).index
            q2_1 = circuit.find_bit(inst2.qubits[1]).index
            
            q3_0 = circuit.find_bit(inst3.qubits[0]).index
            q3_1 = circuit.find_bit(inst3.qubits[1]).index
            
            # Check the SWAP pattern: CX(a, b), CX(b, a), CX(a, b).
            if (q1_0 == q3_0 and q1_1 == q3_1 and  # First and third gates match.
                q2_0 == q1_1 and q2_1 == q1_0):     # The second gate is reversed.
                swap_count += 1
                print(f"  Detected SWAP pattern at position {i}: q{q1_0} <-> q{q1_1}")
                i += 3  # Skip these three gates.
                continue
        
        i += 1
    
    print("\n" + "=" * 70)
    print("Detect CNOT pattern")
    print("=" * 70)
    print(f"Detected SWAP count: {swap_count}")
    
    return swap_count
