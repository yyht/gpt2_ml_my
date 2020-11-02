import os
import re
import time
import tensorflow as tf
from predict.odps_reader import OdpsTableReader
from predict.odpe_writer import OdpsTableWriter


def get_reader_fn():
    """ Automatically  get ez_transfer's reader for different env
    """
    return OdpsTableReader


def get_writer_fn():
    """ Automatically  get ez_transfer's writer for different env
    """
    return OdpsTableWriter

def get_all_columns_name(input_glob):
    """ Get all of the column names for ODPSTable

        Args:
            input_glob (`str`): odps path of the input table.
            Eg. odps://pai_exp_dev/tables/ez_transfer_toy_train



        Returns:
            result (`set`): A set of column names
    """
    reader = tf.python_io.TableReader(input_glob,
                                      selected_cols="",
                                      excluded_cols="",
                                      slice_id=0,
                                      slice_count=1,
                                      num_threads=0,
                                      capacity=0)
    schemas = reader.get_schema()
    return set([col_name for col_name, _, _ in schemas])


def get_selected_columns_schema(input_glob, selected_columns):
    """ Get all of the column schema for the selected columns for ODPSTable

        Args:
            input_glob (`str`): odps path of the input table
            selected_columns (`set`): A set of column names of the input table

        Returns:
            result (`str`): A string of easy transfer defined input schema
    """
    reader = tf.python_io.TableReader(input_glob,
                                      selected_cols="",
                                      excluded_cols="",
                                      slice_id=0,
                                      slice_count=1,
                                      num_threads=0,
                                      capacity=0)
    schemas = reader.get_schema()
    colname2schema = dict()
    for col_name, odps_type, _ in schemas:
        if odps_type == u"string":
            colname2schema[str(col_name)] = "str"
        elif odps_type == u"double":
            colname2schema[str(col_name)] = "float"
        elif odps_type == u"bigint":
            colname2schema[str(col_name)] = "int"
        else:
            colname2schema[str(col_name)] = "str"

    col_with_schemas = ["{}:{}:1".format(col_name, colname2schema[col_name])
                        for col_name in selected_columns if col_name]

    rst_schema = ",".join(col_with_schemas)
    return rst_schema