import numpy as np

from ninatool.spectrum_analysis.spectrum_analysis import HarmonicDiagonalization
from ninatool.internal.elements import J, L
from ninatool.internal.structures import loop
from ninatool.internal.tools import unitsConverter

from scipy.constants import h, e
Phi0 = h / (2 * e)
hbar = h / (2.0 * np.pi)
JtoGHz = 10**(-9) / h



class TestHarmonicDiagonalization:
    @classmethod
    def setup_class(cls):
        EJ = 3.0
        EL = 2.0
        flux = 0.0
        order = 5
        J0 = J(ic=EJ, order=order, name="J0")
        L0 = L(L=1/EL, order=order, name="L0")
        unitconverter = unitsConverter(current_units=1e-6)
        CJ_SI = 10.0  # fF
        CJ = unitconverter.convert_from_fF_to_NINA(CJ_SI)
        left_elements = [J0, ]
        right_elements = [L0, ]
        myrfsquid = loop(
            left_branch=left_elements,
            right_branch=right_elements,
            stray_inductance=False,
            name="myrfsquid"
        )
        spanning_tree = ["J0", ]
        coordination_matrix = np.array([[1, ],
                                        [-1, ]])
        capacitance_matrix = np.array([[CJ, ]])
        node_vars_to_phase_vars = np.array([[1, ], ])
        cls.CJ_SI = CJ_SI
        cls.EC = 1 / CJ
        cls.EJ = EJ
        cls.EL = EL
        cls.harm_diag = HarmonicDiagonalization(
            capacitance_matrix,
            coordination_matrix,
            spanning_tree,
            myrfsquid,
            node_vars_to_phase_vars,
            flux,
            unit_converter=unitconverter,
        )

    def test_energy_to_GHz(self):
        EC_1 = JtoGHz * 10**15 * (e**2 / (2 * self.CJ_SI))
        EC_2 = self.harm_diag.unit_converter.energy_NINA_to_GHz(self.EC)
        assert EC_1 == EC_2

    def test_Xi(self):
        Xi = self.harm_diag.Xi_matrix()[0, 0]
        assert Xi == (8 * self.EC / (self.EJ + self.EL))**(1 / 4)

    def test_omega(self):
        omega_sq, _ = self.harm_diag.eigensystem_normal_modes()
        omega = np.sqrt(omega_sq)[0]
        test_E = omega * self.harm_diag.unit_converter.omega_units * hbar * JtoGHz
        true_E = self.harm_diag.unit_converter.energy_NINA_to_GHz(np.sqrt(8 * self.EC * (self.EJ + self.EL)))
        assert test_E == true_E
