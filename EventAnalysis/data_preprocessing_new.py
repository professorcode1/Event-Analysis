import datetime
import numpy as np
import pandas as pd
from copy import deepcopy
import geopandas as gpd
from shapely.geometry import Point
import multiprocessing 
import re
from tqdm import tqdm


__all__ = ['numpy_from_csv_data', 'get_Df_From_numpy']


def remove_missing_values(dataFile, number_of_rows, viable_column_index):
    absent_entries = set()
    data = np.empty((number_of_rows-1, len(viable_column_index)),dtype=np.float64)
    print("Starting main Loop :: Remove Missing Values")
    rowIndex = 0
    for line in tqdm(dataFile):
        rowIndex += 1
        rowthLine = line
        valuesString = rowthLine.split(',')
        for index,columnIndex in enumerate(viable_column_index):
            if valuesString[columnIndex] == "NA" or valuesString[columnIndex] == ('"NA"'):
                absent_entries.add((rowIndex-1,index))
                data[rowIndex - 1,index] = 0 # when aggregating we will ignore these absent values altogether
                # Normally this would have some central tendency, but the number of entries absent are 0.149667837%
                # So we can ignore them safely
            else:
                data[rowIndex - 1,index] = float(valuesString[columnIndex])
    print(f"Number of values dropped :: {len(absent_entries)}")
    return data, absent_entries

def inside_map_elmnt(element, map_shp, regX_object):
    if regX_object.match(element):
        [x,y] = [float(x) for x in element[1:].split('Y')]
        pnt = Point(x, y)
        return map_shp.contains(pnt).any() or map_shp.touches(pnt).any()
    else:
        return False

def drop_useless_points(columns, map_shp, regX_object, print_dropped_element):
    pool = multiprocessing.Pool()
    print("Dropping Not Point elements / Points outside shape")
    filter_hlpr = pool.starmap(inside_map_elmnt, [(ele, map_shp, regX_object) for ele in columns])
    viable_column_index = list()
    viable_columns = list()
    dropped_elements_cnt = 0
    for idx,element in enumerate(columns):
        if filter_hlpr[idx]:
            viable_column_index.append(idx)
            viable_columns.append(element)
        else:
            dropped_elements_cnt += 1
            if print_dropped_element:
                print(f"dropped_element {element}")
    print(f"Number of elements dropped :: {dropped_elements_cnt}")
    return viable_columns, viable_column_index
            
def numpy_from_csv_data(file_name, map_shape_object, regX_patter = "^X[0-9]+(\.[0-9]+)?Y[0-9]+(\.[0-9]+)?$", remove_double_quotes = True, print_dropped_element = False):
    number_of_rows = sum(1 for line in open(file_name,'r') if line.rstrip()) #first line aka first row is the row with co-ordinates
    dataFile = open(file_name,"r")
    firstLine = dataFile.readline()
    if remove_double_quotes:
        columns = list(map(lambda x : x.rstrip()[1:-1],firstLine.split(','))) #removing the first and last element of x since it
        #are stored in string format. Like "X68.125Y6.125" and the starting and ending " need to be removed
    else:
        columns = list(map(lambda x : x.rstrip(),firstLine.split(',')))
    
    viable_columns, viable_column_index = drop_useless_points(columns, map_shape_object, re.compile(regX_patter), print_dropped_element)
    numpy_mat, missing_vals_index_set = remove_missing_values(dataFile, number_of_rows, viable_column_index)
    return numpy_mat, missing_vals_index_set, viable_columns

def get_Df_From_numpy(numpy_mat,  viable_columns, starting_date = None, time_delta = None, datetime_list = None):
    """
        Create a pandas dataframe to represent the data. The Matrix of the data on grid points, with the time series is used to make the data frame.
        
        If the time series is provided as a python list of datetime objects then it is directly used, if it is not then the starting date and time_delta are 
        used to generate it. 
        
        If both are provided, datatime_list will be used  
    """
    assert (starting_date and time_delta) or datetime_list, "Not enough data provided to create a data frame from the matrix"
    rows, _ = numpy_mat.shape
    if not datetime_list:
        datetime_list = [starting_date + i * time_delta for i in range(rows)]
    df = pd.DataFrame(numpy_mat, columns = viable_columns, index = datetime_list)
    df.index.name = "DateTime"
    return df