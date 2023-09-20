import numpy as np

pi = 3.141592653589793
h = 6.62607015e-34
elementary_charge = 1.602176634e-19

Phi0 = h / (2 * elementary_charge)


class unitsConverter:
    
    def __init__(self, current_units=1e-6):
        self.__current_units = current_units
    
    @property
    def current_units(self):
        return self.__current_units
    
    @property
    def inductance_units(self):
        return Phi0 / (2 * pi * self.current_units)
    
    @property
    def energy_units(self):
        return Phi0 * self.current_units / (2 * pi)
    
    @property
    def capacitance_units(self):
        return pi * elementary_charge ** 2 / (Phi0 * self.current_units)
    
    @property
    def frequency_units(self):
        return self.energy_units / h

    @property
    def impedance_units(self):
        return np.sqrt(self.inductance_units / self.capacitance_units)

    def convert_from_fF_to_NINA(self, capacitance_in_fF):
        return capacitance_in_fF * 10 ** (-15) / self.capacitance_units
    
    @current_units.setter
    def current_units(self, value):
        self.__current_units = value