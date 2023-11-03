# ninatool

The 'Nonlinear Inductive Network Analyzer' (NINA) tool is a python package to analyze 
the low-energy properties of superconducting circuits, based on the theory presented 
in the paper [**"Hamiltonian extrema of an arbitrary Josephson circuit"**](https://arxiv.org/abs/2302.03155).
A YouTube video on the theory behind NINA can be found at the [**IBM Qiskit seminar series #114**](https://www.youtube.com/watch?v=GSswjtMTv6Y).

The main functionality of NINA is to compute the Taylor expansion coefficient of the 
effective potential energy function of an arbitrary flux-biased superconducting loop. 
The loop can host any combination of Josephson junctions (JJs) and linear inductances.
NINA can also compute the Hamiltonian series expansion of an arbitrary Josephson 
nonlinear oscillator (limited for now to a single mode).

NINA includes a simple GUI to allow the user to quickly test the properties of a desired 
superconducting structure (branches and single loops are currently supported, 
more general structures will be available in the future).

# units

NINA uses dimensionless units for electrical variables.

Typically, the user would first fix the desired current units $I_\mathrm{U}$, then 
the inductance units $L_\mathrm{U}$ 
energy units $E_\mathrm{U}$
frequency units $f_\mathrm{U}$
and capacitance units $C_\mathrm{U}$
can be derived from the current units as:

$L_\mathrm{U} = \dfrac{\Phi_0}{2\pi I_\mathrm{U}}$

$E_\mathrm{U} = \dfrac{\Phi_0 I_\mathrm{U}}{2\pi}$

$f_\mathrm{U} = \dfrac{E_\mathrm{U}}{h}$

$C_\mathrm{U} = \dfrac{\pi e^2}{\Phi_0 I_\mathrm{U}}$

where 
$\Phi_0 \approx 2.067 \times 10^{-15}\text{ }\mathrm{Wb}$ is the magnetic flux quantum and 
$e \approx 1.6 \times 10^{-19}\text{ }\mathrm{C}$ is the electron charge.

Phase units are in radians.

Flux is in units of $\dfrac{\Phi_0}{2\pi}$, so can be considered as a phase.

With these units, the following relations hold:

 - Critical current $I_C$ of a JJ corresponds to its Josephson energy $E_J$.

 - Critial current $I_C$ of a JJ and its Josephson inductance $L_J$ are reciprocal.

 - Capacitance $C$ and its charging energy $E_C$ are reciprocal.

# installation instructions

NINA requires the following packages:

- numpy (mandatory)
- sympy (mandatory)
- PyQt5 (for GUI)
- pyqtgraph (for GUI)
- jupyter (for examples)
- matplotlib (for examples)

If you install NINA in a conda environment, make sure to have the required packages
installed via 'conda install [package_name]'.

To install NINA, execute "pip install ." in the package directory.

It is important to not install NINA as a .egg 
(i.e. DON'T execute "python setup.py install"),
as it will fail to load some necessary .txt files.

# learning NINA

Before using NINA, its useful to have a grasp of [**this article**](https://arxiv.org/abs/2302.03155). It will clarify what the terms **free**, **associated branch** etc. mean in NINA.
Code is commented (except for the GUI, for now) and some tutorials will be published soon in the repository.
You can start exploring NINA's functionalities running the .ipynb files in 'examples'.

# citing NINA

If you use NINA for your research, be a good citizen and cite [**this article**](https://arxiv.org/abs/2302.03155).

# contacts

For inquiries, comments, suggestions etc. you can contact the authors at **superconducting.nina@gmail.com**.

# acknowledgements

NINA uses some GUI widgets from [pyqt-labutils](https://github.com/OE-FET/pyqt-labutils) repo.
