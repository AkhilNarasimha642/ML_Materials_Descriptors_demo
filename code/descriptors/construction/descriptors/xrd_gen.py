import numpy as np
from pymatgen.analysis.diffraction.xrd import XRDCalculator


def xrd(pymat_struct, descriptor_size=300, two_theta_min=0.0, two_theta_max=90.0):
    """construct XRD descriptor"""
    # get values from XRD
    two_theta = np.array(XRDCalculator().get_xrd_pattern(pymat_struct).x)
    intensity = np.array(XRDCalculator().get_xrd_pattern(pymat_struct).y)

    # create & digitize bins, include the right bourndary only (zeroth bin is designated for zero only)
    # bin_size = 90/vector_size
    # bins = np.arange(two_theta_min, two_theta_max + bin_size, bin_size)
    bins = np.linspace(two_theta_min, two_theta_max, descriptor_size)
    index_of_bins_for_points = np.digitize(two_theta, bins, right=True)

    # number of points from XRD = len(two_theta) = len(index_of_bins_for_points)
    num_points = len(two_theta)
    descriptor_vector = np.zeros(len(bins))

    # put XRD points into bins
    for i in range(num_points):
        descriptor_vector[index_of_bins_for_points[i]] += intensity[i]
    return descriptor_vector
