import pandas as pd
import numpy as np
__all__ = [
    "createGPUFunction",
    "printTime",
    "create_grid_metadata_arrays",
    "NUMBER_OF_SECONDS_PER_HOUR"
]
NUMBER_OF_SECONDS_PER_HOUR = 3600
def create_grid_metadata_arrays(date_index: pd.DatetimeIndex, rain_event_matrix:np.ndarray):
    
    maximum_number_of_events = -1
    number_of_time_steps, number_of_event_series = rain_event_matrix.shape
    grid_to_event_map_py:list[list[int]] = []
    starting_date = date_index[0]
    grdPnt_numOfEvent:list[int] = []
    date_index_array = np.asarray(list(map(lambda x:x.total_seconds(), ((date_index - starting_date) // 3600))))
    for event_serie_index in range(number_of_event_series):
        this_event_arr:list[int] = []
        for time_step in range(number_of_time_steps):
            if rain_event_matrix[time_step, event_serie_index]:
                this_event_arr.append(date_index_array[time_step])
        grid_to_event_map_py.append(this_event_arr)
        grdPnt_numOfEvent.append(len(this_event_arr))
        maximum_number_of_events = max(maximum_number_of_events, len(this_event_arr))
    for event_serie_index in range(number_of_event_series):
        grid_to_event_map_py[event_serie_index].extend([
            -1 for _ in range(maximum_number_of_events - grdPnt_numOfEvent[event_serie_index])
        ])
    return np.asarray(grid_to_event_map_py, dtype=np.int32), np.asarray(grdPnt_numOfEvent,dtype=np.int32), maximum_number_of_events

def createGPUFunction(sourceCode:str, functionName:str, block:tuple[int,int,int]):
    from pycuda.compiler import SourceModule # type: ignore 
    compilableSrc = sourceCode.replace("BLOCKDIM_X", str(block[0])).replace("BLOCKDIM_Y", str(block[1])).replace("BLOCKDIM_Z", str(block[2]))
    Mod = SourceModule(compilableSrc, no_extern_c = True)
    return Mod.get_function(functionName)
    
def printTime(time_spent:int):
    minutes_spent = time_spent // 60
    seconds_spent = round( time_spent - 60 * minutes_spent)
    print(f"Time elapsed to run the computation :: {minutes_spent} minutes {seconds_spent} seconds ")




