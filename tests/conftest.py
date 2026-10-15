"""Gemeinsame Fixtures: die Mini-Instanz, an der jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def mini():
    """3 x 3-Matrix aus Stück 1 (symmetrisch, Zeilensumme 1, Eigenwerte 1, 0,5, 0,1), Wahrheit x = (1, 2, 1), Daten b = A x = (1,3, 1,4, 1,3).
    Handrechnung mit lambda = 0,1: Filterfaktoren f = 1/1,01, 0,25/0,26, 0,01/0,02 = 0,990099, 0,961538, 0,5. Wegen x = 4/3 (1,1,1) - 1/3 (1,-2,1) ist die Lösung
    ohne Rauschen 0,990099 * 4/3 (1,1,1) - 0,5 * 1/3 (1,-2,1) = (1,153465, 1,653465, 1,153465); die Bias-Norm ist 0,408888 (16,7 % von ||x|| = 2,449490); die
    Rauschvarianz ist sigma^2 * (1/1,01^2 + 0,25/0,26^2 + 0,01/0,02^2) = 29,6785 sigma^2 (ohne Strafe 105 sigma^2); die Freiheitsgrade sind 2,451637."""
    a = np.array([[0.6, 0.3, 0.1], [0.3, 0.4, 0.3], [0.1, 0.3, 0.6]])
    x = np.array([1.0, 2.0, 1.0])
    b = np.array([1.3, 1.4, 1.3])
    return a, x, b, 0.1
