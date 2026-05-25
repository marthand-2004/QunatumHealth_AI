"""
Limitations Module for QuantumHealthAI v2 Enhanced.

This module provides standardized disclaimer strings and an ethics disclaimer
that are appended to every Prediction_Response. It ensures patients and
clinicians are always informed of the system's known limitations, including
dataset heterogeneity, limited clinical data, and the simulated nature of
the quantum circuit.

Satisfies Requirements 11.1, 11.2, 11.3, and 22.1.
"""

STANDARD_LIMITATIONS = [
    "Predictions are based on datasets that may not represent all demographic groups equally.",
    "Training data is limited in size and may not capture rare clinical presentations.",
    "Quantum predictions are produced by a classical simulation of a quantum circuit and do not run on physical quantum hardware.",
]

ETHICS_DISCLAIMER = (
    "This system is NOT a diagnostic tool. Predictions are for clinical decision support only "
    "and must be validated by a licensed medical professional."
)


def get_limitations() -> list[str]:
    """Return a copy of the standard limitations list.

    Returns a copy rather than the original list to prevent callers from
    accidentally mutating the module-level constant.

    Returns:
        list[str]: A list of three standardized disclaimer strings.
    """
    return STANDARD_LIMITATIONS.copy()


def get_disclaimer() -> str:
    """Return the ethics disclaimer string.

    Returns:
        str: The non-dismissible ethics disclaimer for clinical decision support.
    """
    return ETHICS_DISCLAIMER
