import numpy as np
from pymatgen.analysis.local_env import VoronoiNN
from pymatgen.core import periodic_table


def dict_shell_v_electrons_for_one_hot_vector():
    """create a dictionary for one hot vector representation of valence electrons"""
    # s1 - f14 is notated as shell_v_electons
    # number in s1 ... f14 indicates how many valence electrons are in that shell,
    # ex: p5 indicates that there are 5 valence electrons in the p shell
    # value in the dict corresponds to index of the one-hot-vector
    D = {'s1': 0,
         's2': 1}

    p = ['p' + str(i) for i in range(1, 7)]
    d = ['d' + str(i) for i in range(1, 11)]
    f = ['f' + str(i) for i in range(1, 15)]
    p = p + d + f

    for i in range(2, len(p) + 2):
        D[p[i - 2]] = i
    return D


def get_shell_v_electrons(element):
    """get valence electrons for an element in the structure"""
    # s1 - f14 is notated as shell_v_electons
    # get shell_v_electons for corresponding element
    if element == 'H':
        return ['s1']
    elif element == 'He':
        return ['s2']
    else:
        valence_electrons = periodic_table.Element(element).electronic_structure[5:].replace('</sup>.', ' ').replace(
            '</sup>', ' ').split()
        shell_v_electrons_ls = []
        for i in range(len(valence_electrons)):
            valence_electrons[i] = valence_electrons[i].split('<sup>')
            # take the letter of the shell + the correspoinding number of valence e-, ignore principle quantum number
            shell_electron = valence_electrons[i][0][1] + valence_electrons[i][1]
            shell_v_electrons_ls.append(shell_electron)
        return shell_v_electrons_ls


def one_hot_vector_one_element(element):
    """create one hot vector for each element in the structure"""
    # in one-hot-vector notaiton 1 corresponds to the presence of valence e's in the shell
    # according to the key in the dict, for key notation see notes on dict_shell_v_electrons_for_one_hot_vector()
    D = dict_shell_v_electrons_for_one_hot_vector()
    shell_v_electrons_ls = get_shell_v_electrons(element)
    one_hot_vector = np.zeros(len(D))

    for shell_e in shell_v_electrons_ls:
        one_hot_vector[D[shell_e]] = 1
    return one_hot_vector


def dict_one_hot_vectors_all_elements(structure):
    # one-hot-vectors for all elements in the structure
    # key: element, value: one-hot-vector
    valence_vect_dict = {}
    for element in structure.symbol_set:
        valence_vect_dict[element] = one_hot_vector_one_element(element)
    return valence_vect_dict


def get_polyhedras_for_all_sites(structure):
    """get polyhedras for each of the sites in the structure
    return a list"""
    # index of list = index of the structure site
    # given an index returns a list of dictionaries
    # index correspods to the site of the central atom and dictionary has all neighbors as key and corresponding angle ratios as values
    v_polyhedra_sites = []
    for i in range(structure.num_sites):
        v_polyhedra_sites.append(VoronoiNN(allow_pathological=True).get_voronoi_polyhedra(structure, i))
    return v_polyhedra_sites


def ofm(pymat_struct):
    """creates a matrix average of the local environment, then returns the vector repr of the matrix"""
    # pymat_struct = cif_to_pymatgen_structure(cif_path)
    # create a dictionary of element: one-hot-vector shell_v_electrons representation
    valence_vect_dict = dict_one_hot_vectors_all_elements(pymat_struct)
    # go throught each site in the structure and get get_voronoi_polyhedra for each index correspods to the site of
    # the central atom and dictionary has all neighbors as key and corresponding angle ratios as values
    polyhedra_at_each_site = get_polyhedras_for_all_sites(pymat_struct)

    neighbors_total = np.zeros(32)
    descriptors = np.zeros([32, 32])
    # list in the order according to the index of polyhedra_at_each_site
    central_atom_list = [str(pymat_struct.sites[i].specie) for i in range(pymat_struct.num_sites)]
    for central_atom_index in range(pymat_struct.num_sites):
        # central atom vector
        central_atom_vect = np.array(valence_vect_dict[central_atom_list[central_atom_index]])
        # neighbors vector
        for neighbor in polyhedra_at_each_site[central_atom_index]:
            neighbor_vect = np.array(valence_vect_dict[str(neighbor.specie)])
            angle_ratio = polyhedra_at_each_site[central_atom_index][neighbor]
            # calulate vector between the central atom and the neighbor
            distance_vect = neighbor.coords - pymat_struct.sites[central_atom_index].coords
            inverse_distance = (np.sum(distance_vect ** 2)) ** (-0.5)
            neighbor_vect = neighbor_vect * angle_ratio * inverse_distance
            neighbors_total += neighbor_vect
        # descriptor
        descriptors += np.tensordot(neighbors_total, central_atom_vect, 0)
    return (descriptors / pymat_struct.num_sites).flatten()
