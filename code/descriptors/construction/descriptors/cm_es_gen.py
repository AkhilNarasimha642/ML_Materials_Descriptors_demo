from pymatgen.analysis.ewald import EwaldSummation
from scipy.linalg import eigvalsh
import numpy as np


def cm_es(pymat_struct, zeros_up_to = 2000):
    """note: function pads constructed descr with zeros up to chosen zeros_up_to. Function
    df_CM_ES_tayloring in create_descriptors.py taylors it to the max non zero index in the
    given dataset"""    
    pymat_struct.add_site_property("charge",pymat_struct.atomic_numbers)
    energy_matrix = EwaldSummation(pymat_struct).total_energy_matrix
    e_val_vector = eigvalsh(energy_matrix)
    descr_with_pad_zeros = np.zeros(zeros_up_to)
    descr_with_pad_zeros[:len(e_val_vector)] = e_val_vector
    return descr_with_pad_zeros
