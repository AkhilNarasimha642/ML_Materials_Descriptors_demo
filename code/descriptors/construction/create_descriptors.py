import os
import numpy as np
import pandas as pd
from argparse import ArgumentParser

from pymatgen.io.cif import CifParser
from pymatgen.io.vasp import Poscar

from descriptors.xrd_gen import xrd
from descriptors.ofm_gen import ofm
from descriptors.cm_es_gen import cm_es

parser = ArgumentParser(description='Chose descriptor and file batch name')
parser.add_argument('database', type=str)
parser.add_argument('descriptor_name', type=str)
args = parser.parse_args()


def file_to_pymat_struct(struct_path):
    """import cif or POSCAR convert to pymatgen file"""
    if struct_path[-3:] == 'cif':
        parser = CifParser(struct_path)
        return parser.get_structures()[0]
    elif struct_path[-6:] == 'POSCAR':
        poscar = Poscar.from_file(struct_path)
        return poscar.structure
    else:
        raise ValueError('Please provide cif or POSCAR structure')


def get_root(num_layers=3):
    """find the root directory of the project"""
    curren_dir = os.path.dirname(os.path.realpath(__file__))
    for i in range(num_layers):
        curren_dir = os.path.dirname(curren_dir)
    return curren_dir


def df_cm_es_tayloring(df):
    """find largest index for nonzero df features val and taylor df = df[:ind+1]"""
    oridinal_len = len(df.index)
    largest_ind = -1
    for name in df.columns:
        descr = df[name].values
        for i in range(oridinal_len - 1, largest_ind, -1):
            if descr[i] > 0 and i > largest_ind:
                largest_ind = i
                break
    largest_ind += 1
    return df[:largest_ind]


def descriptors_full(database, descriptor_name):
    """construct CM or OFM or XRD descripor out of files in the provided database
    save the output as df"""
    comp_path = get_root()
    structures_path = comp_path + '/data/' + database + '/structure_files'  # either .cif or .POSCAR
    all_struct_files_names = os.listdir(structures_path)
    descriptors_dict = {'xrd': xrd, 'cm_es': cm_es, 'ofm': ofm}
    descriptor = descriptors_dict[descriptor_name]
    descriptor_len = len(
        descriptor(file_to_pymat_struct(structures_path + '/' + all_struct_files_names[0])))  # test descr
    descriptor_mat = np.zeros([len(all_struct_files_names), descriptor_len])
    c = 0
    for i in range(len(all_struct_files_names)):
        struct_file_name = all_struct_files_names[i]
        pymat_struct = file_to_pymat_struct(structures_path + '/' + struct_file_name)
        descriptor_mat[i][:] = descriptor(pymat_struct)
        c += 1
    descriptor_mat = np.round(descriptor_mat, 6)
    save_path = comp_path + '/data/' + database + '/descriptor_dfs/' + descriptor_name
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    df = pd.DataFrame(descriptor_mat.T, columns=all_struct_files_names)
    if descriptor_name == 'CM_ES':
        df = df_cm_es_tayloring(df)
    df.to_csv(save_path + '/' + descriptor_name + '.csv')


def create_descriptors(database, descriptor_name):
    """creates df of the descriptor out of a given the database of files"""
    if descriptor_name in ('xrd', 'cm_es', 'ofm'):
        descriptors_full(database, descriptor_name)
    print(descriptor_name + ' df created')


if __name__ == '__main__':
    create_descriptors(args.database, args.descriptor_name)
