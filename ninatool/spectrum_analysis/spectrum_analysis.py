import numpy as np
import scipy as sp
from numpy import ndarray
from scipy.constants import h, e
from scipy.linalg import eigh
from sympy import S, symbols, factorial, sqrt, exp, expand, degree, degree_list
from qutip import destroy, qeye, tensor, Qobj


from ninatool.internal.structures import loop
from ninatool.internal.elements import L, J, C
from ninatool.circuits.base_circuits import snail
from ninatool.internal.tools import unitsConverter

Phi0 = h / (2 * e)
hbar = h / (2.0 * np.pi)
JtoGHz = 10**(-9) / h


def id_wrap_ops(op: Qobj, idx: int, truncated_dims: list) -> Qobj:
    """
    identity wrap the operator op which has index idx in a system
    where each subsystem has Hilbert dim as specified by truncated_dims
    Parameters
    ----------
    op: Qobj
        single subsystem operator
    idx: int
        position of the subsystem
    truncated_dims: list
        Hilbert space dimension of the subsystems

    Returns
    -------
    Qobj
    """
    assert op.dims[0][0] == truncated_dims[idx]
    id_list = [qeye(dim) for dim in truncated_dims]
    id_list[idx] = op
    return tensor(*id_list)


class HarmonicDiagonalization:
    """
    Class for diagonalizing the harmonic part of the Hamiltonian
    and then using the normal modes to calculate the anharmonic terms

    Parameters
    ----------
    capacitance_matrix: ndarray
        capacitance matrix expressed in the node variables
    coordination_matrix: ndarray
        coordination matrix of the circuit relating the elements in the potential
        to the node variables. The rows correspond to the elements and the columns
        to the node variables. The entries should be 0, 1 or -1 to indicate the form of
        the potential element. For example for an element of the form -EJ * cos(\phi_2 - \phi_0), the corresponding
        row would look like -1, 0, 1
    spanning_tree: list
        list of the elements in the loop_instance that form the spanning tree
    loop_instance: loop
        circuit specified in the NINA way
    node_vars_to_phase_vars: ndarray
        matrix specifying the transformation between the node and phase variables
    flux: float
    """

    def __init__(
        self,
        capacitance_matrix: ndarray,
        coordination_matrix: ndarray,
        spanning_tree: list,
        loop_instance: loop,
        node_vars_to_phase_vars: ndarray,
        flux: float,
        unit_converter: unitsConverter,
    ):
        self.capacitance_matrix = capacitance_matrix
        self.coordination_matrix = coordination_matrix
        self.spanning_tree = spanning_tree
        self.loop_instance = loop_instance
        self.node_vars_to_phase_vars = node_vars_to_phase_vars
        self.flux = flux
        self.num_modes = self.capacitance_matrix.shape[0]
        char_list = "a b c d e f g h i j k l m n o p q r s t u v w x y z".split()
        char_dag_list = [char + "\u2020" for char in char_list]
        self.lowering_ops = symbols(char_list[0: self.num_modes])
        self.raising_ops = symbols(char_dag_list[0: self.num_modes])
        self.unit_converter = unit_converter

    def find_minimum_node_variables(self) -> ndarray:
        self.loop_instance.interpolate_results(2.0 * np.pi * self.flux)
        minimum_loc_difference_phases = np.array(
            [
                self.loop_instance.elements[idx].phi[0]
                for idx, elem in enumerate(self.loop_instance.elements)
                if elem.name in self.spanning_tree
            ]
        )
        return (
            sp.linalg.inv(self.node_vars_to_phase_vars) @ minimum_loc_difference_phases
        )

    def gamma_matrix(self) -> ndarray:
        """Returns linearized potential matrix

        We are imagining an arbitrary loop of JJs where we have
        changed variables to the difference variables, so that
        each junction is a function of just one variable, except for
        the last junction, which is a function of all of the variables

        Returns
        -------
        ndarray
        """
        dim = self.capacitance_matrix.shape[0]
        minimum_location = self.find_minimum_node_variables()
        gamma_matrix = np.zeros((dim, dim))
        # below assumes that the rows of the coordination matrix correspond to
        # the entries in self.loop_instance.elements
        for node_idx, node_var_spec in enumerate(self.coordination_matrix):
            # the below relationship holds bc we are using NINA units
            # (EJ and Ic are the same and Ic and LJ are reciprocal)
            Ic = self.loop_instance.elements[node_idx].ic
            EJ = Ic * self.unit_converter.current_units * hbar * JtoGHz / (2 * e)
            inv_inductance = EJ / 0.25  # units where hbar=e=1, so Phi0 = 0.5
            if isinstance(self.loop_instance.elements[node_idx], J):

                def _inductance_func(equilibrium_phase_):
                    return np.cos(equilibrium_phase_)
            elif isinstance(self.loop_instance.elements[node_idx], L):

                def _inductance_func(equilibrium_phase_):
                    return 1
            else:
                raise RuntimeError(
                    "should only have inductors and junctions in the potential"
                )
            nonzero_idxs = np.argwhere(node_var_spec)[:, 0]
            equilibrium_phase = minimum_location @ node_var_spec
            if self.loop_instance.elements[node_idx].name not in self.spanning_tree:
                equilibrium_phase = equilibrium_phase - 2.0 * np.pi * self.flux
            if len(nonzero_idxs) == 1:  # only a single node variable
                gamma_matrix[
                    nonzero_idxs[0], nonzero_idxs[0]
                ] += inv_inductance * _inductance_func(equilibrium_phase)
            elif (
                len(nonzero_idxs) == 2
            ):  # in this case two node variables, so get off-diag elements
                gamma_matrix[
                    nonzero_idxs[0], nonzero_idxs[0]
                ] += inv_inductance * _inductance_func(equilibrium_phase)
                gamma_matrix[
                    nonzero_idxs[1], nonzero_idxs[1]
                ] += inv_inductance * _inductance_func(equilibrium_phase)
                gamma_matrix[
                    nonzero_idxs[0], nonzero_idxs[1]
                ] += -inv_inductance * _inductance_func(equilibrium_phase)
                gamma_matrix[
                    nonzero_idxs[1], nonzero_idxs[0]
                ] += -inv_inductance * _inductance_func(equilibrium_phase)
            else:
                raise RuntimeError("each branch should only be connected to two nodes")
        return gamma_matrix

    def eigensystem_normal_modes(self) -> (ndarray, ndarray):
        """Returns squared normal mode frequencies, matrix of eigenvectors

        Parameters
        ----------
        minimum_index: int
            integer specifying which minimum to linearize around,
            0<=minimum<= total number of minima

        Returns
        -------
        ndarray, ndarray
        """
        omega_squared, normal_mode_eigenvectors = eigh(
            self.gamma_matrix(), b=self.capacitance_matrix
        )
        return omega_squared, normal_mode_eigenvectors

    def Xi_matrix(self) -> ndarray:
        """Returns Xi matrix of the normal-mode eigenvectors normalized
        according to \Xi^T C \Xi = \Omega^{-1}/Z0, or equivalently \Xi^T
        \Gamma \Xi = \Omega/Z0. The \Xi matrix
        simultaneously diagonalizes the capacitance and effective
        inductance matrices by a congruence transformation.

        Parameters
        ----------
        None

        Returns
        -------
        ndarray
        """
        omega_squared_array, eigenvectors = self.eigensystem_normal_modes()
        Z0 = 0.25
        Ximat = np.array(
            [
                eigenvectors[:, i] * omega_squared ** (-1 / 4) * np.sqrt(1.0 / Z0)
                for i, omega_squared in enumerate(omega_squared_array)
            ]
        ).T
        assert np.allclose(Ximat.T @ self.capacitance_matrix @ Ximat, np.diag(omega_squared_array**(-1/2)) / Z0)
        return Ximat

    def normal_ordered_kinetic(self):
        dim = self.capacitance_matrix.shape[0]
        op_list = list(zip(self.lowering_ops[0:dim], self.raising_ops[0:dim]))
        omega_squared, _ = self.eigensystem_normal_modes()
        kin = S(0)
        for idx, omega_sq in enumerate(omega_squared):
            op, op_dag = op_list[idx]
            omega = S(np.sqrt(omega_sq))
            # one factor of 1/2 from split between potential and kinetic, other factor
            # of 1/2 from definition of n in terms of a, adag
            kin += -S(0.25) * omega * (op_dag * op_dag + op * op - S(2) * op_dag * op)
        return kin

    def _normal_ordered_kinetic_test(self):
        dim = self.capacitance_matrix.shape[0]
        op_list = list(zip(self.lowering_ops[0:dim], self.raising_ops[0:dim]))
        kin = S(0)
        Xi = self.Xi_matrix()
        Xi_inv = sp.linalg.inv(Xi)
        transformed_EC = Xi_inv @ sp.linalg.inv(self.capacitance_matrix) @ Xi_inv.T / 2
        for idx_1 in range(dim):
            for idx_2 in range(dim):
                op_1, op_1_dag = op_list[idx_1]
                op_2, op_2_dag = op_list[idx_2]
                kin += -4 * transformed_EC[idx_1, idx_2] * (op_1 - op_1_dag) * (op_2 - op_2_dag)
        return self._filter_small_coeffs_and_high_order(expand(kin), threshold=1e-9, order=4)

    def normal_ordered_potential(self, threshold=1e-8, order=5):
        dim = self.capacitance_matrix.shape[0]
        op_list = list(zip(self.lowering_ops[0:dim], self.raising_ops[0:dim]))
        pot = S(0)
        Xi = self.Xi_matrix()
        for node_idx, node_var_spec in enumerate(self.coordination_matrix):
            Ic = self.loop_instance.elements[node_idx].ic
            EJ = Ic * self.unit_converter.current_units * hbar * JtoGHz / (2 * e)
            nonzero_idxs = np.argwhere(node_var_spec)[:, 0]
            normal_mode_prefactors = np.sum(
                [Xi[idx, :] * node_var_spec[idx] for idx in nonzero_idxs], axis=0
            )
            if isinstance(self.loop_instance.elements[node_idx], J):
                if self.loop_instance.elements[node_idx].name not in self.spanning_tree:
                    pot += -EJ * (self.expand_cosine(
                        normal_mode_prefactors, op_list, order=order
                    ) * np.cos(2.0 * np.pi * self.flux) + self.expand_sine(
                        normal_mode_prefactors, op_list, order=order
                    ) * np.sin(
                        2.0 * np.pi * self.flux
                    ))
                else:
                    pot += -EJ * self.expand_cosine(normal_mode_prefactors, op_list, order=order)
            elif isinstance(self.loop_instance.elements[node_idx], L):
                pot += -EJ * self.expand_cosine(normal_mode_prefactors, op_list, order=2)
            else:
                raise RuntimeError("should only have inductors and junctions in the potential")
        pot = self._filter_small_coeffs_and_high_order(expand(pot), threshold=threshold, order=order)
        return pot

    @staticmethod
    def _filter_small_coeffs_and_high_order(symp_expr, threshold=1e-8, order=5):
        terms = symp_expr.as_ordered_terms()
        new_term = S(0)
        for idx, term in enumerate(terms):
            if not term.is_Float:
                if abs(term.args[0]) >= threshold and sum(degree_list(term)) <= order:
                    new_term += term
        return new_term

    def _expanded_sines(self, prefactors, op_strs, order):
        last_pref = [prefactors[-1], ]
        last_op = [op_strs[-1], ]
        sin_allm1 = self.expand_sine(
            prefactors[0:-1], op_strs[0:-1], order=order
        )
        cos_1 = self.expand_cosine(
            last_pref, last_op, order=order,
        )
        cos_allm1 = self.expand_cosine(
            prefactors[0:-1], op_strs[0:-1], order=order
        )
        sin_1 = self.expand_sine(
            last_pref, last_op, order=order,
        )
        return sin_allm1, cos_allm1, cos_1, sin_1

    def expand_sine(self, prefactors, op_strs, order=5):
        if len(prefactors) == len(op_strs) == 1:
            xi = S(prefactors[0])
            op, op_dag = op_strs[0]
            return sum(
                [
                    exp(-(xi**2) / S(4))
                    * (xi / S(np.sqrt(2)))
                    * (-(xi**2) / S(2)) ** S((n + m - 1) / 2)
                    / (factorial(S(n)) * factorial(S(m)))
                    * op_dag ** S(n)
                    * op ** S(m)
                    for n in range(0, order + 1)
                    for m in range(0, order + 1)
                    if (n+m) % 2 == 1 and n + m <= order
                ]
            )
        else:
            sin_allm1, cos_allm1, cos_1, sin_1 = self._expanded_sines(
                prefactors, op_strs, order
            )
            return sin_allm1 * cos_1 + cos_allm1 * sin_1

    def expand_cosine(self, prefactors, op_strs, order=5):
        if len(prefactors) == len(op_strs) == 1:
            xi = S(prefactors[0])
            op, op_dag = op_strs[0]
            # cos[(xi / sqrt(2)) * (a + a^{\dag})]
            return sum(
                [
                    exp(-(xi**2) / S(4))
                    * (-(xi**2) / S(2)) ** S((n+m)/2)
                    / (factorial(S(n)) * factorial(S(m)))
                    * op_dag ** S(n)
                    * op ** S(m)
                    for n in range(0, order + 1)
                    for m in range(0, order + 1)
                    if (n + m) % 2 == 0 and n + m <= order
                ]
            )
        else:
            sin_allm1, cos_allm1, cos_1, sin_1 = self._expanded_sines(
                prefactors, op_strs, order
            )
            return cos_allm1 * cos_1 - sin_allm1 * sin_1

    def hamiltonian_from_sympy(self, sym_H, cutoff=3):
        sym_H = sym_H.as_ordered_terms()
        a = destroy(cutoff)
        annihilation_ops = [id_wrap_ops(a, idx, self.num_modes * [cutoff])
                            for idx in range(self.num_modes)]
        total_H = 0.0
        for sym_H_term in sym_H:
            H_term = complex(sym_H_term.args[0])
            for idx, sym_create_op in enumerate(self.raising_ops):
                create_degree = degree(sym_H_term, gen=sym_create_op)
                H_term = H_term * annihilation_ops[idx].dag() ** int(create_degree)
            for idx, sym_lower_op in enumerate(self.lowering_ops):
                create_degree = degree(sym_H_term, gen=sym_lower_op)
                H_term = H_term * annihilation_ops[idx] ** int(create_degree)
            total_H += H_term
        return total_H


# musnail
if __name__ == "__main__":
    flux = 0.3
    alpha = 0.2
    musnail = snail()
    spanning_tree = ["J1", "J2", "J3"]
    coordination_matrix = np.array([[1, 0, 0], [-1, 1, 0], [0, -1, 1], [0, 0, -1]])
    unitconverter = unitsConverter(current_units=1e-6)
    EC = 0.2
    ECJ = 2.0
    C = 1 / (2 * EC)
    CJ = 1 / (2 * ECJ)
    capacitance_matrix = np.array([[C + 2 * CJ, -CJ, 0], [-CJ, 2 * CJ, -CJ], [0, -CJ, 2 * CJ]])
    node_vars_to_phase_vars = coordination_matrix[0:3, :]
    harm_diag = HarmonicDiagonalization(
        capacitance_matrix,
        coordination_matrix,
        spanning_tree,
        musnail,
        node_vars_to_phase_vars,
        flux,
        unit_converter=unitconverter
    )
    result_pot = harm_diag.normal_ordered_potential(order=4)
    H = harm_diag.hamiltonian_from_sympy(result_pot)
    result_kin = harm_diag.normal_ordered_kinetic()
    result_kin_test = harm_diag._normal_ordered_kinetic_test()


if __name__ == "__main__":
    flux = 0.0
    order = 5
    J0 = J(ic=40000.0, order=order, name="J0")
    L0 = L(L=1000000.0, order=order, name="L0")
    unitconverter = unitsConverter(current_units=1e-9)
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
    # CJ_SI = 10.0  # fF
    # ECJ = e**2 / (2 * CJ_SI * 10**(-15)) * JtoGHz
    ECJ = 2.0  # GHz
    capacitance_matrix = np.array([[1 / (2 * ECJ), ]])
    node_vars_to_phase_vars = np.array([[1, ], ])
    harm_diag = HarmonicDiagonalization(
        capacitance_matrix,
        coordination_matrix,
        spanning_tree,
        myrfsquid,
        node_vars_to_phase_vars,
        flux,
        unit_converter=unitconverter,
    )
    result = harm_diag.normal_ordered_potential(order=3)
    result_kin = harm_diag.normal_ordered_kinetic()
    print(0)



if __name__ == "__main__":
    flux = 0.0
    snail = snail()
    spanning_tree = ["J1", "J2", "J3"]
    coordination_matrix = np.array([[1, 0, 0], [-1, 1, 0], [0, -1, 1], [0, 0, -1]])
    # CJ_SI = 10.0  # fF
    # C_SI = 100.0  # fF
    unitconverter = unitsConverter(current_units=1e-8)
    # CJ = unitconverter.convert_from_fF_to_NINA(CJ_SI)
    # C = unitconverter.convert_from_fF_to_NINA(C_SI)
    EC = 0.2
    ECJ = 2.0
    C = 1 / (2 * EC)
    CJ = 1/ (2 * ECJ)
    capacitance_matrix = np.array([[C + 2 * CJ, -CJ, 0], [-CJ, 2 * CJ, -CJ], [0, -CJ, 2 * CJ]])
    node_vars_to_phase_vars = coordination_matrix[0:3, :]
    harm_diag = HarmonicDiagonalization(
        capacitance_matrix,
        coordination_matrix,
        spanning_tree,
        snail,
        node_vars_to_phase_vars,
        flux,
        unit_converter=unitconverter,
    )
    result = harm_diag.normal_ordered_potential(order=3)
    print(0)
