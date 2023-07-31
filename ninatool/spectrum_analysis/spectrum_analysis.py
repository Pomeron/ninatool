import numpy as np
import scipy as sp
from numpy import ndarray
from scipy.linalg import eigh
from sympy import S, symbols, factorial, sqrt, exp
from sympy.physics.secondquant import B, Dagger

from ninatool.internal.structures import loop
from ninatool.internal.elements import L, J
from ninatool.circuits.base_circuits import snail


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
            flux: float
    ):
        self.capacitance_matrix = capacitance_matrix
        self.coordination_matrix = coordination_matrix
        self.spanning_tree = spanning_tree
        self.loop_instance = loop_instance
        self.node_vars_to_phase_vars = node_vars_to_phase_vars
        self.flux = flux
        char_list = 'a b c d e f g h i j k l m n o p q r s t u v w x y z'.split()
        self.lowering_ops = symbols(char_list)
        op_list = [char+r"$^\dagger$"+char for char in char_list]
        self.op_list = symbols(op_list)

    def find_minimum_node_variables(self) -> ndarray:
        self.loop_instance.interpolate_results(2.0 * np.pi * self.flux)
        minimum_loc_difference_phases = np.array([
            self.loop_instance.elements[idx].phi[0] for idx, elem in enumerate(self.loop_instance.elements)
            if elem.name in self.spanning_tree
        ])
        return sp.linalg.inv(self.node_vars_to_phase_vars) @ minimum_loc_difference_phases

    def gamma_matrix(self) -> ndarray:
        """Returns linearized potential matrix

        Note that we must divide by Phi_0^2 since Ej/Phi_0^2 = 1/Lj,
        or one over the effective impedance of the junction.

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
        # how to ensure ordering is the same between elements and
        # entries in coordination_matrix?
        for idx, node_var_spec in enumerate(self.coordination_matrix):
            inv_inductance = self.loop_instance.elements[idx].ic
            if isinstance(self.loop_instance.elements[idx], J):
                def _inductance_func(equilibrium_phase_):
                    return np.cos(equilibrium_phase_)
            elif isinstance(self.loop_instance.elements[idx], L):
                def _inductance_func(equilibrium_phase_):
                    return 1
            else:
                raise RuntimeError("should only have inductors and junctions in the potential")
            nonzero_idxs = np.argwhere(node_var_spec)[0]
            equilibrium_phase = minimum_location @ node_var_spec
            if self.loop_instance.elements[idx].name not in self.spanning_tree:
                equilibrium_phase = equilibrium_phase - 2.0 * np.pi * self.flux
            if len(nonzero_idxs) == 1:  # only a single node variable
                gamma_matrix[nonzero_idxs[0], nonzero_idxs[0]] += inv_inductance * _inductance_func(equilibrium_phase)
            elif len(node_var_spec) == 2:  # in this case two node variables, so get off-diag elements
                gamma_matrix[nonzero_idxs[0], nonzero_idxs[0]] += inv_inductance * _inductance_func(equilibrium_phase)
                gamma_matrix[nonzero_idxs[1], nonzero_idxs[1]] += inv_inductance * _inductance_func(equilibrium_phase)
                gamma_matrix[nonzero_idxs[0], nonzero_idxs[1]] += -inv_inductance * _inductance_func(equilibrium_phase)
                gamma_matrix[nonzero_idxs[1], nonzero_idxs[0]] += -inv_inductance * _inductance_func(equilibrium_phase)
            else:
                raise RuntimeError("each branch should only be connected to two nodes")
        Phi0 = 0.5  # units where e_charge, hbar = 1; Phi0 = hbar / (2 * e)
        return gamma_matrix / Phi0**2

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
        Z0 = 0.25  # units where e and hbar = 1; Z0 = hbar / (2 * e)**2
        return np.array(
            [
                eigenvectors[:, i]
                * omega_squared ** (-1 / 4)
                * np.sqrt(1.0 / Z0)
                for i, omega_squared in enumerate(omega_squared_array)
            ]
        ).T

    # should be trivial to include kinetic
    def normal_ordered_potential(self, order=5):
        dim = self.capacitance_matrix.shape[0]
        op_list = self.lowering_ops[0:dim]
        pot = S(0)
        Xi = self.Xi_matrix()
        for idx, node_var_spec in enumerate(self.coordination_matrix):
            nonzero_idxs = np.argwhere(node_var_spec)[0]
            # abusing notation here a little: of course in the cosine,
            # the operator isn't a^{\dagger}a
            normal_mode_prefactors = np.sum([Xi[idx, :] * node_var_spec[idx] for idx in nonzero_idxs], axis=0)
            if isinstance(self.loop_instance.elements[idx], J):
                if self.loop_instance.elements[idx].name not in self.spanning_tree:
                    pot += (self.expand_cosine(normal_mode_prefactors, op_list, order=order)
                            * np.cos(2.0 * np.pi * self.flux)
                            + self.expand_sine(normal_mode_prefactors, op_list, order=order)
                            * np.sin(2.0 * np.pi * self.flux)
                            )
                else:
                    pot += self.expand_cosine(normal_mode_prefactors, op_list, order=order)
            elif isinstance(self.loop_instance.elements[idx], L):
                pot += self.expand_cosine(normal_mode_prefactors, op_list, order=2)
            else:
                raise RuntimeError("should only have inductors and junctions in the potential")
        return pot

    def expand_sine(self, normal_mode_prefactors, ladder_op_strs, order=5):
        if len(normal_mode_prefactors) == len(ladder_op_strs) == 1:
            eta = S(normal_mode_prefactors[0])
            op = ladder_op_strs[0]
            return sum([exp(-eta / 4) * sqrt(eta / 2)
                        * (-eta/2) ** (S(2 * S(ord_) - 1)/2) / factorial(S(ord_))**2 * op ** S(ord_)
                        for ord_ in range(1, order + 1, 2)])
        else:
            return (self.expand_sine(normal_mode_prefactors[0: -1], ladder_op_strs[0: -1], order=order) *
                    self.expand_cosine([normal_mode_prefactors[-1], ], [ladder_op_strs[-1], ], order=order)
                    + self.expand_cosine(normal_mode_prefactors[0: -1], ladder_op_strs[0: -1], order=order) *
                    self.expand_sine([normal_mode_prefactors[-1], ], [ladder_op_strs[-1], ], order=order)
                    )

    def expand_cosine(self, normal_mode_prefactors, ladder_op_strs, order=5):
        if len(normal_mode_prefactors) == len(ladder_op_strs) == 1:
            # simplification here of making RWA: also of course get counter-rotating terms,
            # may want to be included in the future
            eta = S(normal_mode_prefactors[0])
            op = ladder_op_strs[0]
            return sum([exp(-eta / 4) * (-eta / 2) ** S(ord_) / factorial(S(ord_)) ** 2 * op ** S(ord_)
                        for ord_ in range(0, order + 1, 2)])
        else:
            return (self.expand_cosine(normal_mode_prefactors[0: -1], ladder_op_strs[0: -1], order=order) *
                    self.expand_cosine([normal_mode_prefactors[-1], ], [ladder_op_strs[-1], ], order=order)
                    - self.expand_sine(normal_mode_prefactors[0: -1], ladder_op_strs[0: -1], order=order) *
                    self.expand_sine([normal_mode_prefactors[-1], ], [ladder_op_strs[-1], ], order=order)
            )


if __name__ == "__main__":
    flux = 0.3
    snail = snail()
    spanning_tree = ["J1", "J2", "J3"]
    coordination_matrix = np.array([[1, 0, 0],
                                   [-1, 1, 0],
                                   [0, -1, 1],
                                   [0, 0, -1]])
    CJ = 1.0
    C = 1.0
    capacitance_matrix = np.array([[C + 2 * CJ, -CJ, 0],
                                   [-CJ, 2 * CJ, -CJ],
                                   [0, -CJ, 2 * CJ]])
    node_vars_to_phase_vars = coordination_matrix[0:3, :]
    harm_diag = HarmonicDiagonalization(capacitance_matrix, coordination_matrix, spanning_tree,
                                        snail, node_vars_to_phase_vars, flux)
    result = harm_diag.normal_ordered_potential(order=5)
    print(0)
