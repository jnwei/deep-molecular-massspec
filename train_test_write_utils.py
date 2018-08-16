"""Contains functions to help write dataset files for train test splits.

Contains helper functions to help write TFRecords, info files, and inchikey
files for a list of inchikeys, and a dict of a list of mols keyed by inchikey.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import os
import random

from absl import app
from absl import flags
import parse_sdf_utils
import train_test_split_utils
import tensorflow as tf

INCHIKEY_FILENAME_END = '.inchikey.txt'
TFRECORD_FILENAME_END = '.tfrecord'
NP_LIBRARY_ARRAY_END = '.spectra_library.npy'
FROM_MAINLIB_FILENAME_MODIFIER = '_from_mainlib'
FROM_REPLICATES_FILENAME_MODIFIER = '_from_replicates'

def write_list_of_inchikeys(inchikey_list, base_name, output_dir):
  """Write list of inchikeys as a text file."""
  inchikey_list_name = base_name + INCHIKEY_FILENAME_END

  with tf.gfile.Open(os.path.join(output_dir, inchikey_list_name),
                     'w') as writer:
    for inchikey in inchikey_list:
      writer.write('%s\n' % inchikey)


def write_all_dataset_files(inchikey_dict,
                            inchikey_list,
                            base_name,
                            output_dir,
                            max_atoms,
                            max_mass_spec_peak_loc,
                            make_library_array=False):
  """Helper function for writing all the files associated with a TFRecord.

  Args:
    inchikey_dict : Full dictionary keyed by inchikey containing lists of
                    rdkit.Mol objects
    inchikey_list : List of inchikeys to include in dataset
    base_name : Base name for the dataset
    output_dir : Path for saving all TFRecord files
    max_atoms : Maximum number of atoms to include for a given molecule
    max_mass_spec_peak_loc : Largest m/z peak to include in a spectra.
    make_library_array : Flag for whether to make library array
  Returns:
    Saves 3 files:
     basename.tfrecord : a TFRecord file,
     basename.inchikey.txt : a text file with all the inchikeys in the dataset
     basename.tfrecord.info: a text file with one line describing
         the length of the TFRecord file.
    Also saves if make_library_array is set:
     basename.npz : see parse_sdf_utils.write_dicts_to_example
  """
  record_name = base_name + TFRECORD_FILENAME_END

  mol_list = train_test_split_utils.make_mol_list_from_inchikey_dict(
      inchikey_dict, inchikey_list)

  if make_library_array:
    library_array_pathname = base_name + NP_LIBRARY_ARRAY_END
    parse_sdf_utils.write_dicts_to_example(
        mol_list, os.path.join(output_dir, record_name),
        max_atoms, max_mass_spec_peak_loc,
        os.path.join(output_dir, library_array_pathname))
  else:
    parse_sdf_utils.write_dicts_to_example(
        mol_list, os.path.join(output_dir, record_name), max_atoms,
        max_mass_spec_peak_loc)
  write_list_of_inchikeys(inchikey_list, base_name, output_dir)
  parse_sdf_utils.write_info_file(mol_list, os.path.join(
      output_dir, record_name))


def write_datasets_from_lib(component_inchikey_dict, lib_inchikey_dict,
                            output_dir, max_atoms, max_mass_spec_peak_loc,
                            lib_name='mainlib'):
  """Write all train/val/test set TFRecords and info from NIST mainlibrary."""
  if lib_name == 'mainlib':
    filename_modifier = FROM_MAINLIB_FILENAME_MODIFIER
  elif lib_name == 'replicates':
    filename_modifier = FROM_REPLICATES_FILENAME_MODIFIER

  for component_kwarg in component_inchikey_dict.keys():
    component_lib_filename = (
        component_kwarg + filename_modifier)
    write_all_dataset_files(
        lib_inchikey_dict,
        component_inchikey_dict[component_kwarg],
        component_lib_filename,
        output_dir,
        max_atoms,
        max_mass_spec_peak_loc)