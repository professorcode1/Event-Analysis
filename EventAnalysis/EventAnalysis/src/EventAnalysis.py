
from numba import njit,jit, prange
import numba
import numpy as np
import math
import pandas as pd
import os
import time
from .ECAKernel import * 
from .ESKernel import * 
from .ECALoop import * 
from .ESLoop import * 
from .utils import * 
from datetime import timedelta

__all__ = ['EventAnalysis']


class EventAnalysis:
    def __init__(self, event_df_:pd.DataFrame, device_Id:None|int = None, time_normalization_factor = NUMBER_OF_SECONDS_PER_HOUR):
        #As defined in Event synchrony measures for functional climate network analysis: A case study on South American rainfall dynamics
        event_df = event_df_.sort_index()
        date_index = event_df.index
        self.date_index = date_index
        self.time_normalization_factor = time_normalization_factor
        assert isinstance(date_index, pd.DatetimeIndex), "the index of the event_df must be of type pandas DatetimeIndex"
        self.coordinate_columns = event_df.columns
        rain_event_matrix = event_df.to_numpy(copy = True)

        assert rain_event_matrix.dtype == bool, "The rain event dataframe does not have bool as its internal type, are you sure you sent an event series and not a time series?"
        
        starting_date = date_index[0]
        ending_date = date_index[-1]
        self.T = ( ending_date - starting_date ).total_seconds() // self.time_normalization_factor
        assert self.T <= np.iinfo(np.int32).max,f'the time span of data provided({self.T}) is over what a 32bit int can store {np.iinfo(np.int32).max}. Erroring now to prevent returning incorrect resuts' 
        self.number_of_grid_points = rain_event_matrix.shape[1]

        self.grid_to_event_map, \
            self.grdPnt_numOfEvent, \
            self.maximum_number_of_events =\
            create_grid_metadata_arrays(date_index, rain_event_matrix, self.time_normalization_factor)

        self.logedFactorial = np.empty(self.maximum_number_of_events + 1, dtype = np.float64)
        self.logedFactorial[ 0 ] = 0
        for lg_fc_i in np.arange(1, self.maximum_number_of_events + 1):
            self.logedFactorial[ lg_fc_i ] = self.logedFactorial[ lg_fc_i - 1 ] + np.log(lg_fc_i)
        
        self.cudaInitialised = False
        self.device_Id = device_Id

    def initialiseCuda(self):
        self.cudaInitialised = True
        if not self.device_Id:
            print("Device ID was not specified, reverting to 0")
            self.device_Id = 0  
        
        import pycuda.driver as cuda # type: ignore 
        from pycuda.compiler import SourceModule # type: ignore 
        cuda.init()
        self.ES_Source = ES_CODE
        self.ECA_Source = ECA_CODE
        self.ECA_Pval_Source = PVALUE_CODE

        self.ES_block_cnfg = 16, 8, 1
        self.ECA_block_cnfg = 32, 1, 1

        self.grid = (self.number_of_grid_points, self.number_of_grid_points, 1)

        self.device = cuda.Device(self.device_Id)
        print(f"Device in use \n\t{self.device.name()}")
        self.device.make_context()

        self.ECA_GPU = createGPUFunction(self.ECA_Source, ECA_kernel_name , self.ECA_block_cnfg)
        self.ECA_Pval_GPU = createGPUFunction(self.ECA_Pval_Source, P_value_kernel_name , self.ECA_block_cnfg)
        self.ES_GPU = createGPUFunction(self.ES_Source, ES_kernel_name , self.ES_block_cnfg)

        self.grdPnt_numOfEvent_GPU_ = cuda.mem_alloc(self.grdPnt_numOfEvent.nbytes)
        cuda.memcpy_htod(self.grdPnt_numOfEvent_GPU_, self.grdPnt_numOfEvent)

        self.grid_to_event_GPU_ = cuda.mem_alloc(self.grid_to_event_map.nbytes)
        cuda.memcpy_htod(self.grid_to_event_GPU_, self.grid_to_event_map)
        
        self.logedFactorial_GPU_ = cuda.mem_alloc(self.logedFactorial.nbytes)
        cuda.memcpy_htod(self.logedFactorial_GPU_, self.logedFactorial)

        self.num_of_grids_GPU_ = np.array(self.number_of_grid_points, dtype=np.int32)
        self.max_nm_events_GPU_ = np.array(self.maximum_number_of_events, dtype=np.int32)

    def ES(self, tauMax:float = float('inf')):

        assert tauMax == float('inf'), "Currently the code doesn't use tauMax. Please open an issue and notify the author that it is required and it will be added ASAP!"
        Tau_hlpr_left = np.roll(self.grid_to_event_map, -1, axis=1) - self.grid_to_event_map
        Tau_hlpr_right = self.grid_to_event_map - np.roll(self.grid_to_event_map, 1, axis=1)
        Tau_hlpr = np.minimum(Tau_hlpr_left, Tau_hlpr_right)
        for grid_point, number_of_events in enumerate(self.grdPnt_numOfEvent):
            Tau_hlpr[grid_point, 0] = Tau_hlpr_left[grid_point, 0]
            Tau_hlpr[grid_point,number_of_events - 1] = Tau_hlpr_right[grid_point,number_of_events - 1]
        # To better utilise numpy vectorisation, Tau_hlpr is used. It is defined as
        # Tau_hlpr[i][l] = min(t^{i}_{l+1} - t^{i}_{l},t^{i}_{l} - t^{i}_{l-1})
        # except at l= 0 and l = si - 1 , since those are the corner cases
        c_i_j = np.zeros((self.number_of_grid_points, self.number_of_grid_points), dtype=np.float32)

        start_time = time.time()
        ES_Loop(self.number_of_grid_points, self.grdPnt_numOfEvent, Tau_hlpr, tauMax, self.grid_to_event_map, c_i_j)
        end_time = time.time()
        time_spent = end_time - start_time
        printTime(time_spent)
        
        Q = np.add(c_i_j, c_i_j.T).astype('float32')
        normaliseQMatrix(Q, self.grdPnt_numOfEvent)
        df = pd.DataFrame(Q, index = self.coordinate_columns, columns = self.coordinate_columns)
        df.index.name = "Coordinates"
        return df
    
    def ECA(self, Delta_T_obj:timedelta, tau:int = 0, return_p_values = False, pValFillNA = True):

        r_precursor_i_j = np.zeros((self.number_of_grid_points, self.number_of_grid_points), dtype = np.float32)
        r_trigger_i_j = np.zeros((self.number_of_grid_points, self.number_of_grid_points), dtype = np.float32)

        Delta_T = Delta_T_obj.total_seconds() // self.time_normalization_factor
        
        time_spent = 0
        start_time = time.time()
        
        ECA_Loop(self.number_of_grid_points, self.grdPnt_numOfEvent, self.grid_to_event_map, Delta_T, tau, r_precursor_i_j, r_trigger_i_j)
        end_time = time.time()
        time_spent += end_time - start_time

        EC_p_max  = pd.DataFrame(np.maximum(r_precursor_i_j, r_precursor_i_j.T), index = self.coordinate_columns , columns = self.coordinate_columns)
        EC_p_mean = pd.DataFrame( ( r_precursor_i_j + r_precursor_i_j.T ) / 2,   index = self.coordinate_columns , columns = self.coordinate_columns)
        EC_t_max  = pd.DataFrame(np.maximum(r_trigger_i_j, r_trigger_i_j.T),     index = self.coordinate_columns , columns = self.coordinate_columns)
        EC_t_mean = pd.DataFrame( ( r_trigger_i_j + r_trigger_i_j.T ) / 2,       index = self.coordinate_columns , columns = self.coordinate_columns)

        EC_p_max.index.name  = "Coordinates"
        EC_p_mean.index.name = "Coordinates"
        EC_t_max.index.name  = "Coordinates"
        EC_t_mean.index.name = "Coordinates"

        return_value = None
        if return_p_values:
            pval_precursor = np.zeros((self.number_of_grid_points, self.number_of_grid_points), dtype = np.float64)
            pval_trigger = np.zeros((self.number_of_grid_points, self.number_of_grid_points), dtype = np.float64)
            start_time = time.time()
            
            find_p_value_new(self.number_of_grid_points, self.grdPnt_numOfEvent, Delta_T, tau, r_precursor_i_j, r_trigger_i_j, pval_precursor, pval_trigger, self.T, self.logedFactorial)


            end_time = time.time()
            time_spent += end_time - start_time

            if(pValFillNA):
                pval_precursor = np.nan_to_num(pval_precursor, nan = 1.0)
                pval_trigger = np.nan_to_num(pval_trigger, nan = 1.0)
            
            return_value = EC_p_max, EC_p_mean, EC_t_max, EC_t_mean, pval_precursor, pval_trigger
        else:
            return_value = EC_p_max, EC_p_mean, EC_t_max, EC_t_mean
        printTime(time_spent)
        return return_value

    def ECA_vec(self, Delta_T_objs:list[timedelta], taus:int|np.ndarray|list[int] = 0, return_p_values = False, pValFillNA = True):
        numr_results = len(Delta_T_objs)
        if type(taus) == int:
            taus = np.full(numr_results, taus, dtype = np.int32)
        elif type(taus) == list:
            assert len(taus) == len(Delta_T_objs), "There is a different b/w the numebr of taus and number of DeltaT objects" 
            taus = np.asarray(taus, dtype = np.int32)
        elif isinstance(taus, np.ndarray):
            assert taus.ndim == 1 and taus.shape[0] == numr_results and taus.dtype == np.int32, f"tau should be an ndarray of ints of shape ({numr_results}, )"
        else :
            raise Exception("Unrecognised arguement for taus, it must either be a scalar int, or a list/ndarray of ints")

        time_spent = 0
        for res in np.arange(numr_results):
            Delta_T = int(Delta_T_objs[res].total_seconds() // self.time_normalization_factor)

            tau = taus[res]
            r_precursor_i_j = np.zeros((self.number_of_grid_points, self.number_of_grid_points), dtype = np.float32)
            r_trigger_i_j = np.zeros((self.number_of_grid_points, self.number_of_grid_points), dtype = np.float32)
            
            start_time = time.time()
            
            ECA_Loop(self.number_of_grid_points, self.grdPnt_numOfEvent, self.grid_to_event_map, Delta_T, tau, r_precursor_i_j, r_trigger_i_j)
            
            end_time = time.time()
            time_spent += end_time - start_time
            
            EC_p_max  = pd.DataFrame(np.maximum(r_precursor_i_j, r_precursor_i_j.T), index = self.coordinate_columns , columns = self.coordinate_columns)
            EC_p_mean = pd.DataFrame( ( r_precursor_i_j + r_precursor_i_j.T ) / 2,   index = self.coordinate_columns , columns = self.coordinate_columns)
            EC_t_max  = pd.DataFrame(np.maximum(r_trigger_i_j, r_trigger_i_j.T),     index = self.coordinate_columns , columns = self.coordinate_columns)
            EC_t_mean = pd.DataFrame( ( r_trigger_i_j + r_trigger_i_j.T ) / 2,       index = self.coordinate_columns , columns = self.coordinate_columns)

            EC_p_max.index.name  = "Coordinates"
            EC_p_mean.index.name = "Coordinates"
            EC_t_max.index.name  = "Coordinates"
            EC_t_mean.index.name = "Coordinates"

            if return_p_values:
                pval_precursor = np.zeros((self.number_of_grid_points, self.number_of_grid_points), dtype = np.float64)
                pval_trigger = np.zeros((self.number_of_grid_points, self.number_of_grid_points), dtype = np.float64)
            
                start_time = time.time()
            
                find_p_value_new(self.number_of_grid_points, self.grdPnt_numOfEvent, Delta_T, tau, r_precursor_i_j, r_trigger_i_j, pval_precursor, pval_trigger, self.T, self.logedFactorial)
                
                end_time = time.time()
                time_spent += end_time - start_time
        
                if(pValFillNA):
                    pval_precursor = np.nan_to_num(pval_precursor, nan = 1.0)
                    pval_trigger = np.nan_to_num(pval_trigger, nan = 1.0)
        
                yield (EC_p_max, EC_p_mean, EC_t_max, EC_t_mean, pval_precursor, pval_trigger)
            else:
                yield (EC_p_max, EC_p_mean, EC_t_max, EC_t_mean)
        printTime(time_spent)

    def ES_Cuda(self, tauMax = float('inf'), block = None):
        assert tauMax == float('inf'), "Currently the code doesn't use tauMax. Please open an issue and notify the author that it is required and it will be added ASAP!"
        import pycuda.driver as cuda# type: ignore 
        from pycuda.compiler import SourceModule# type: ignore 
        from pycuda._driver import device_attribute# type: ignore 
        if not self.cudaInitialised:
            self.initialiseCuda()    

        nmbr_thrds_pr_blk = self.device.get_attributes()[device_attribute.MAX_THREADS_PER_BLOCK]
        warp_size = self.device.get_attributes()[device_attribute.WARP_SIZE]
        if block:
            if block != self.ES_block_cnfg :
                assert block[2] == 1 and (block[0] * block[1]) % warp_size ==  0, f"block parameter to this function must be of the the form (x,y,1) where x * y  is a multiple of {warp_size}"
                self.ES_block_cnfg = block
                print("Recompiling code for the new block size")
                self.ES_GPU = createGPUFunction(self.ES_Source, ES_kernel_name , self.ES_block_cnfg)

        C_Gpu = cuda.mem_alloc(4 * self.number_of_grid_points * self.number_of_grid_points)
        C = np.empty((self.number_of_grid_points, self.number_of_grid_points), dtype = np.float32)

        start_time = time.time()

        self.ES_GPU(C_Gpu, self.grdPnt_numOfEvent_GPU_, self.grid_to_event_GPU_, self.num_of_grids_GPU_, self.max_nm_events_GPU_, grid = self.grid, block = self.ES_block_cnfg)
        cuda.memcpy_dtoh(C, C_Gpu)

        end_time = time.time()
        time_spent = end_time - start_time
        printTime(time_spent)

        Q = np.add(C, C.T).astype('float32')
        normaliseQMatrix(Q, self.grdPnt_numOfEvent)
        df = pd.DataFrame(Q, index = self.coordinate_columns, columns = self.coordinate_columns)
        df.index.name = "Coordinates"
        return df

    def ECA_Cuda(self, Delta_T_obj:timedelta, tau:int = 0, return_p_values = False, block = None, pValFillNA = True):
        import pycuda.driver as cuda# type: ignore 
        from pycuda.compiler import SourceModule# type: ignore 
        from pycuda._driver import device_attribute# type: ignore 
        if not self.cudaInitialised:
            self.initialiseCuda()    
        
        nmbr_thrds_pr_blk = self.device.get_attributes()[device_attribute.MAX_THREADS_PER_BLOCK]
        warp_size = self.device.get_attributes()[device_attribute.WARP_SIZE]
        
        if block:
            if block != self.ECA_block_cnfg:
                assert block[1] == 1 and block[2] == 1 and block[0] % warp_size == 0, f"block parameter should be of the form (x,1,1) where x is a multiple of {warp_size}"
                print("Recompiling code for the new block size")
                self.ECA_block_cnfg = block
                self.ECA_GPU = createGPUFunction(self.ECA_Source, ECA_kernel_name , self.ECA_block_cnfg)
                self.ECA_Pval_GPU = createGPUFunction(self.ECA_Pval_Source, P_value_kernel_name , self.ECA_block_cnfg)

        Delta_T_GPU = np.array(Delta_T_obj.total_seconds() // self.time_normalization_factor, dtype=np.int32)
        Tau_GPU = np.array(tau, dtype=np.int32)
        r_precursor_GPU = cuda.mem_alloc(4 * self.number_of_grid_points * self.number_of_grid_points)
        r_trigger_GPU = cuda.mem_alloc(4 * self.number_of_grid_points * self.number_of_grid_points)
        r_precursor_i_j = np.empty((self.number_of_grid_points , self.number_of_grid_points), dtype = np.float32 )
        r_trigger_i_j = np.empty((self.number_of_grid_points , self.number_of_grid_points), dtype = np.float32 )


        time_spent = 0
        start_time = time.time()

        self.ECA_GPU(r_precursor_GPU, r_trigger_GPU, self.grdPnt_numOfEvent_GPU_, self.grid_to_event_GPU_, Delta_T_GPU, Tau_GPU, self.num_of_grids_GPU_, self.max_nm_events_GPU_, grid = self.grid, block = self.ECA_block_cnfg)
        cuda.memcpy_dtoh(r_precursor_i_j, r_precursor_GPU)

        end_time = time.time()
        time_spent += (end_time - start_time)

        cuda.memcpy_dtoh(r_trigger_i_j, r_trigger_GPU)

        EC_p_max  = pd.DataFrame(np.maximum(r_precursor_i_j, r_precursor_i_j.T), index = self.coordinate_columns , columns = self.coordinate_columns)
        EC_p_mean = pd.DataFrame( ( r_precursor_i_j + r_precursor_i_j.T ) / 2,   index = self.coordinate_columns , columns = self.coordinate_columns)
        EC_t_max  = pd.DataFrame(np.maximum(r_trigger_i_j, r_trigger_i_j.T),     index = self.coordinate_columns , columns = self.coordinate_columns)
        EC_t_mean = pd.DataFrame( ( r_trigger_i_j + r_trigger_i_j.T ) / 2,       index = self.coordinate_columns , columns = self.coordinate_columns)

        EC_p_max.index.name  = "Coordinates"
        EC_p_mean.index.name = "Coordinates"
        EC_t_max.index.name  = "Coordinates"
        EC_t_mean.index.name = "Coordinates"
        return_value = None
        if return_p_values:
            pval_precursor = np.zeros((self.number_of_grid_points, self.number_of_grid_points), dtype = np.float64)
            pval_trigger = np.zeros((self.number_of_grid_points, self.number_of_grid_points), dtype = np.float64)
            pval_precursor_gpu = cuda.mem_alloc( self.number_of_grid_points * self.number_of_grid_points * 8 )
            pval_trigger_gpu = cuda.mem_alloc( self.number_of_grid_points * self.number_of_grid_points * 8 )
            T_Gpu = np.array(self.T, dtype = np.int32)
            
            start_time = time.time()
            
            self.ECA_Pval_GPU(pval_precursor_gpu,pval_trigger_gpu,r_precursor_GPU,r_trigger_GPU ,self.grdPnt_numOfEvent_GPU_,Delta_T_GPU,Tau_GPU, self.num_of_grids_GPU_, T_Gpu, self.logedFactorial_GPU_, grid = self.grid, block = self.ECA_block_cnfg)
            cuda.memcpy_dtoh( pval_precursor ,pval_precursor_gpu )
            
            end_time = time.time()
            time_spent += (end_time - start_time)
            cuda.memcpy_dtoh(pval_trigger, pval_trigger_gpu)
            
            if(pValFillNA):
                pval_precursor = np.nan_to_num(pval_precursor, nan = 1.0)
                pval_trigger = np.nan_to_num(pval_trigger, nan = 1.0)

            return_value = EC_p_max, EC_p_mean, EC_t_max, EC_t_mean, pval_precursor, pval_trigger
        else:
            return_value = EC_p_max, EC_p_mean, EC_t_max, EC_t_mean
        printTime(time_spent)
        return return_value

    def ECA_vec_Cuda(self, Delta_T_objs:list[timedelta], taus:int|np.ndarray|list[int] = 0, return_p_values = False, block = None, pValFillNA = True):
        numr_results = len(Delta_T_objs)
        if type(taus) == int:
            taus = np.full(numr_results, taus, dtype = np.int32)
        elif type(taus) == list:
            assert len(taus) == len(Delta_T_objs), "There is a different b/w the numebr of taus and number of DeltaT objects" 
            taus = np.asarray(taus, dtype = np.int32)
        elif isinstance(taus, np.ndarray):
            assert taus.ndim == 1 and taus.shape[0] == numr_results and taus.dtype == np.int32, f"tau should be an ndarray of int32s of shape ({numr_results}, )"
        else :
            raise Exception("Unrecognised arguement for taus, it must either be a scalar int, or a list/ndarray of ints")

        import pycuda.driver as cuda# type: ignore 
        from pycuda.compiler import SourceModule# type: ignore 
        from pycuda._driver import device_attribute# type: ignore 
        if not self.cudaInitialised:
            self.initialiseCuda()    
        nmbr_thrds_pr_blk = self.device.get_attributes()[device_attribute.MAX_THREADS_PER_BLOCK]
        warp_size = self.device.get_attributes()[device_attribute.WARP_SIZE]
        
        if block:
            if block != self.ECA_block_cnfg:
                assert block[1] == 1 and block[2] == 1 and block[0] % warp_size == 0, f"block parameter should be of the form (x,1,1) where x is a multiple of {warp_size}"
                print("Recompiling code for the new block size")
                self.ECA_block_cnfg = block
                self.ECA_GPU = createGPUFunction(self.ECA_Source, ECA_kernel_name , self.ECA_block_cnfg)
                self.ECA_Pval_GPU = createGPUFunction(self.ECA_Pval_Source, P_value_kernel_name , self.ECA_block_cnfg)

        r_precursor_GPU = cuda.mem_alloc(4 * self.number_of_grid_points * self.number_of_grid_points)
        r_trigger_GPU = cuda.mem_alloc(4 * self.number_of_grid_points * self.number_of_grid_points)
        r_precursor_i_j = np.empty((self.number_of_grid_points , self.number_of_grid_points), dtype = np.float32 )
        r_trigger_i_j = np.empty((self.number_of_grid_points , self.number_of_grid_points), dtype = np.float32 )
        time_spent = 0

        pval_precursor_gpu = None
        pval_trigger_gpu = None
        
        if return_p_values:
            pval_precursor_gpu = cuda.mem_alloc( self.number_of_grid_points * self.number_of_grid_points * 8 )
            pval_trigger_gpu = cuda.mem_alloc( self.number_of_grid_points * self.number_of_grid_points * 8 )

        Delta_T_GPU = np.array([Delta_T_obj.total_seconds() // self.time_normalization_factor for Delta_T_obj in Delta_T_objs], dtype=np.int32)
        for res in np.arange(numr_results):
            # print(res)
            
            start_time = time.time()
            self.ECA_GPU(r_precursor_GPU, r_trigger_GPU, self.grdPnt_numOfEvent_GPU_, self.grid_to_event_GPU_, Delta_T_GPU[res], taus[res], self.num_of_grids_GPU_, self.max_nm_events_GPU_, grid = self.grid, block = self.ECA_block_cnfg)
            cuda.memcpy_dtoh(r_precursor_i_j, r_precursor_GPU)
            end_time = time.time()
            time_spent += end_time - start_time

            cuda.memcpy_dtoh(r_trigger_i_j, r_trigger_GPU)

            EC_p_max  = pd.DataFrame(np.maximum(r_precursor_i_j, r_precursor_i_j.T), index = self.coordinate_columns , columns = self.coordinate_columns)
            EC_p_mean = pd.DataFrame( ( r_precursor_i_j + r_precursor_i_j.T ) / 2,   index = self.coordinate_columns , columns = self.coordinate_columns)
            EC_t_max  = pd.DataFrame(np.maximum(r_trigger_i_j, r_trigger_i_j.T),     index = self.coordinate_columns , columns = self.coordinate_columns)
            EC_t_mean = pd.DataFrame( ( r_trigger_i_j + r_trigger_i_j.T ) / 2,       index = self.coordinate_columns , columns = self.coordinate_columns)

            EC_p_max.index.name  = "Coordinates"
            EC_p_mean.index.name = "Coordinates"
            EC_t_max.index.name  = "Coordinates"
            EC_t_mean.index.name = "Coordinates"

            if return_p_values:
                pval_precursor = np.zeros((self.number_of_grid_points, self.number_of_grid_points), dtype = np.float64)
                pval_trigger = np.zeros((self.number_of_grid_points, self.number_of_grid_points), dtype = np.float64)
                T_Gpu = np.array(self.T, dtype = np.int32)

                start_time = time.time()

                self.ECA_Pval_GPU(pval_precursor_gpu,pval_trigger_gpu,r_precursor_GPU,r_trigger_GPU ,self.grdPnt_numOfEvent_GPU_,Delta_T_GPU[res], taus[res], self.num_of_grids_GPU_, T_Gpu,self.logedFactorial_GPU_, grid = self.grid, block = self.ECA_block_cnfg)
                cuda.memcpy_dtoh( pval_precursor ,pval_precursor_gpu )

                end_time = time.time()
                time_spent += (end_time - start_time)
                cuda.memcpy_dtoh(pval_trigger, pval_trigger_gpu)

                if(pValFillNA):
                    pval_precursor = np.nan_to_num(pval_precursor, nan = 1.0)
                    pval_trigger = np.nan_to_num(pval_trigger, nan = 1.0)


                yield (EC_p_max, EC_p_mean, EC_t_max, EC_t_mean, pval_precursor, pval_trigger)
            else:
                yield (EC_p_max, EC_p_mean, EC_t_max, EC_t_mean)
        printTime(time_spent)
