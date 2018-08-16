"""Split the NIST dataset according to families.

This module is used to split the NIST dataset by family type,
then create train/validation splits for each family type.

The components are then saved as individual TFRecord files using
functions in train_test_write_utils. The *.info files and *.inchikey.txt
files are also written, see train_test_write_utils.write_all_dataset files
for details.

The supported family types are recorded in feature_utils.FILTER_DICT
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import os
import random

from absl import app
from absl import flags
import dataset_setup_constants as ds_constants
import mass_spec_constants as ms_constants
import parse_sdf_utils
import train_test_split_utils
import train_test_write_utils
import feature_utils
import tensorflow as tf

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'main_sdf_name', 'testdata/test_14_mend.sdf',
    'specify full path of sdf file to parse, to be used for'
    ' training sets, and validation/test sets')
flags.DEFINE_string(
    'replicates_sdf_name',
    'testdata/test_2_mend.sdf',
    'specify full path of a second sdf file to parse, to be'
    ' used for the vaildation/test set. Molecules in this sdf'
    ' will be excluded from the main train/val/test sets.')
flags.DEFINE_string('output_master_dir', '/tmp/output_dataset_dir',
                    'specify directory to save records')
flags.DEFINE_list(
    'family_name_list', 'diazo',
    'specify list of family names to split the datasets into.'
    'Should be names in feature_utils.FEATURE_DICT')
flags.DEFINE_list(
    'val_test_split', '0.5,0.5',
    'specify list of ratios to use for validation/test split'
    ' in replicates')
flags.DEFINE_integer('max_atoms', ms_constants.MAX_ATOMS,
                     'specify maximum number of atoms to allow')
flags.DEFINE_integer('max_mass_spec_peak_loc', ms_constants.MAX_PEAK_LOC,
                     'specify greatest m/z spectrum peak to allow')


def rename_dict_keys_for_split_name(inchikey_dict, dataset_split_name):
  new_inchikey_dict = dict([(dataset_split_name + '_' + key, value)
    for key, value in inchikey_dict.iteritems()])
  return new_inchikey_dict


def main(_):
  print('begin class train tests split')
  tf.gfile.MkDir(FLAGS.output_master_dir)

  # Read sdf files for molecules
  print('reading sdf files')
  mainlib_mol_list = parse_sdf_utils.get_sdf_to_mol(
  	FLAGS.main_sdf_name)
  replib_mol_list = parse_sdf_utils.get_sdf_to_mol(
  	FLAGS.replicates_sdf_name)
  
  # Convert molecule list into dict. 
  mainlib_mol_dict = train_test_split_utils.make_inchikey_dict(
  	mainlib_mol_list)
  replib_mol_dict = train_test_split_utils.make_inchikey_dict(
  	replib_mol_list)

  mainlib_inchikeys, replib_inchikeys = (
  	train_test_split_utils.make_mainlib_replicates_split(mainlib_mol_dict, replib_mol_dict))

  print('Splitting datasets by family')
  # Split inchikeys by family
  family_name_list = [name for name in FLAGS.family_name_list]
  mainlib_family_dict = train_test_split_utils.make_splits_by_family(
  	mainlib_inchikeys, mainlib_mol_dict, family_name_list)
  mainlib_family_dict = rename_dict_keys_for_split_name(mainlib_family_dict,
   'MAINLIB')

  replib_family_dict = train_test_split_utils.make_splits_by_family(
  	replib_inchikeys, replib_mol_dict, FLAGS.family_name_list)
  replib_family_dict = rename_dict_keys_for_split_name(replib_family_dict,
   'REPLIB')

  # Split replicates into test/validation sets.
  val_test_fractions = tuple(
      [float(elem) for elem in FLAGS.val_test_split])

  replicates_component_dict = (
    train_test_split_utils.split_all_replicate_inchikey_list(
  	replib_family_dict, val_test_fractions))

  print(replicates_component_dict)

  # Write datasets to disc
  train_test_write_utils.write_datasets_from_lib(
    mainlib_family_dict, mainlib_mol_dict,
    FLAGS.output_master_dir, FLAGS.max_atoms,
    FLAGS.max_mass_spec_peak_loc, lib_name='mainlib')

  train_test_write_utils.write_datasets_from_lib(
    replicates_component_dict, mainlib_mol_dict,
    FLAGS.output_master_dir, FLAGS.max_atoms,
    FLAGS.max_mass_spec_peak_loc, lib_name='mainlib')

  train_test_write_utils.write_datasets_from_lib(
    replicates_component_dict, replib_mol_dict,
    FLAGS.output_master_dir, FLAGS.max_atoms,
    FLAGS.max_mass_spec_peak_loc, lib_name='replicates')


if __name__ == '__main__':
	app.run(main)
