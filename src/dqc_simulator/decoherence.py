"""
Idle decoherence from the scheduled timeline.

    Routing -> Scheduling -> Waiting -> Decoherence -> Fidelity

For every idle gap Δt of a qubit (Timeline.instruction_idle), a noise channel is
inserted right before the instruction that ends the gap. T1/T2 are read from the
QPU's Qiskit Target (qubit_properties). Gate-time decoherence is already part of
NoiseModel.from_backend, so only idle gaps are added here (no double counting).
"""

import numpy as np
from qiskit_aer.noise import thermal_relaxation_error, pauli_error


# Fallbacks (seconds) when a backend reports no T1/T2 for a qubit.
DEFAULT_T1 = 100e-6
DEFAULT_T2 = 100e-6

IDLE_NOISE_MODELS = ("thermal", "dephasing")


class DecoherenceModel:
    """
    Convert an idle duration on a physical qubit into a Qiskit noise channel.

    model="thermal"  : thermal_relaxation_error(T1, T2, Δt)   (amplitude + phase damping)
    model="dephasing": Z with p_Z(Δt) = (1 - exp(-Δt / T_phi)) / 2,
                       1/T_phi = 1/T2 - 1/(2 T1)            (reference section 9)
    """

    def __init__(self, model="thermal", default_t1=DEFAULT_T1, default_t2=DEFAULT_T2, min_idle=0.0):
        if model not in IDLE_NOISE_MODELS:
            raise ValueError(f"Unknown idle noise model '{model}'. Available: {IDLE_NOISE_MODELS}")
        self.model = model
        self.default_t1 = default_t1
        self.default_t2 = default_t2
        self.min_idle = min_idle      # ignore gaps shorter than this

    def get_t1_t2(self, qpu, qubit):
        """T1/T2 of a physical qubit from the QPU target, with defaults."""
        t1, t2 = None, None
        target = getattr(qpu, "target", None)
        props = getattr(target, "qubit_properties", None) if target is not None else None
        if props is not None and qubit < len(props) and props[qubit] is not None:
            t1, t2 = props[qubit].t1, props[qubit].t2
        t1 = self.default_t1 if t1 is None else t1
        t2 = self.default_t2 if t2 is None else t2
        return t1, min(t2, 2 * t1)    # physical constraint T2 <= 2 T1

    def get_idle_error(self, qpu, qubit, idle_time):
        """QuantumError for `idle_time` seconds on `qubit` of `qpu`, or None if negligible."""
        if idle_time <= self.min_idle:
            return None
        t1, t2 = self.get_t1_t2(qpu, qubit)
        if self.model == "thermal":
            return thermal_relaxation_error(t1, t2, idle_time)

        rate = 1.0 / t2 - 1.0 / (2.0 * t1)
        if rate <= 0:
            return None
        p_z = (1.0 - np.exp(-idle_time * rate)) / 2.0
        return pauli_error([("Z", p_z), ("I", 1.0 - p_z)])


def add_idle_noise(circuit, timeline, qubit_location, decoherence=None):
    """
    Return a copy of `circuit` with idle-noise channels inserted from `timeline`.

    :param circuit: merged circuit that `timeline` was scheduled from
    :param timeline: Timeline with instruction_idle
    :param qubit_location: {global_qubit: (qpu, physical_qubit)}
    :param decoherence: DecoherenceModel (default: thermal relaxation)
    """
    if decoherence is None:
        decoherence = DecoherenceModel()

    new_circ = circuit.copy_empty_like()
    for i, ci in enumerate(circuit.data):
        for q, idle_time in timeline.instruction_idle.get(i, []):
            qpu, phys = qubit_location[q]
            error = decoherence.get_idle_error(qpu, phys, idle_time)
            if error is not None:
                new_circ.append(error.to_instruction(), [new_circ.qubits[q]])
        new_circ.append(ci.operation, ci.qubits, ci.clbits)
    return new_circ
