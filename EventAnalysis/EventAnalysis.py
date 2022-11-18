
from numba import njit,jit, prange
import numba
import numpy as np
import math
import pandas as pd
import os
import time
number_of_seconds_per_hour = 3600
ECA_kernel_name = "ECA_kernel"
ES_kernel_name = "ES_kernel"
P_value_kernel_name = "pVal_kernel"

__all__ = ['EventAnalysis']

@njit(nogil = True)
def numOfEvents_onGridPoint(point_to_num_map, grid_point):
    if np.where(point_to_num_map[0, :] == grid_point)[0].size:
        return point_to_num_map[1, np.where(point_to_num_map[0, :] == grid_point)[0][0]]
    else:
        return 0

# @njit((numba.int64, numba.int32[::1], numba.int32[:, ::1], numba.float64, numba.int32[:,::1], numba.float64[:,::1]),parallel = True, nogil = True)
@njit(parallel = True, nogil = True)
def ES_Loop(number_of_grid_points, grdPnt_numOfEvent, Tau_hlpr, tauMax, grid_to_event_map, c_i_j):
    for i in prange(number_of_grid_points):
        # if math.floor( i * 100 / number_of_grid_points ) != math.floor( ( i - 1 ) * 100 / number_of_grid_points ):
        #     print(i * 100 / number_of_grid_points)
        s_i = grdPnt_numOfEvent[i]
        if s_i in [0,1,2]:
            continue
        for j in range(number_of_grid_points):
            s_j = grdPnt_numOfEvent[j]
            if s_j in [0,1,2]:
                continue
            for l in range(1, s_i - 1):
                for m in range(1, s_j - 1):
                    t_l = grid_to_event_map[i, l]
                    t_m = grid_to_event_map[j, m]
                    tao = 0.5 * min(Tau_hlpr[i, l], Tau_hlpr[j, m])
                    sigma_i_j_l_m = (t_l - t_m <= tao) and t_l > t_m
                    
                    t_l_minus_one = grid_to_event_map[i, l-1]
                    tao = 0.5 * min(Tau_hlpr[i, l-1], Tau_hlpr[j, m])
                    sigma_j_i_m_l_minus_one = t_l_minus_one < t_m and (t_m - t_l_minus_one <= tao)
                    
                    t_m_plus_one = grid_to_event_map[j, m + 1]
                    tao = 0.5 * min(Tau_hlpr[i, l], Tau_hlpr[j, m + 1])
                    sigma_j_i_m_plus_one_l = t_m_plus_one > t_l and (t_m_plus_one - t_l <= tao)
                    
                    if(sigma_i_j_l_m and not(sigma_j_i_m_l_minus_one or sigma_j_i_m_plus_one_l)):
                        c_i_j[i, j] += 1
                    elif t_l == t_m or (sigma_i_j_l_m and (sigma_j_i_m_l_minus_one or sigma_j_i_m_plus_one_l)):
                        c_i_j[i, j] += 0.5


# @njit((numba.int64, numba.int32[ ::1 ], numba.int32[:, ::1 ], numba.float64, numba.int64, numba.float64[:, ::1 ], numba.float64[:, ::1 ]),parallel=True, nogil= True)
@njit(parallel=True, nogil= True)
def ECA_Loop(number_of_grid_points: np.ndarray, grdPnt_numOfEvent:np.ndarray , grid_to_event_map : np.ndarray, Delta_T : int, tau :int, r_precursor_i_j : np.ndarray, r_trigger_i_j : np.ndarray):
    for i in prange(number_of_grid_points):
        # if math.floor( i * 100 / number_of_grid_points ) != math.floor( ( i - 1 ) * 100 / number_of_grid_points ):
            # print(i*100/number_of_grid_points)
        for j in np.arange(number_of_grid_points):
            numOfEvents_on_i = grdPnt_numOfEvent[i]
            numOfEvents_on_j = grdPnt_numOfEvent[j]

            for l in np.arange(numOfEvents_on_i):
                t_l = grid_to_event_map[i, l]
                bin_ser_low = 0
                bin_ser_high = numOfEvents_on_j - 1
                while bin_ser_low <= bin_ser_high:
                    bin_ser_mid = math.floor( (bin_ser_low + bin_ser_high) / 2 )
                    t_m = grid_to_event_map[j, bin_ser_mid]
                    diff = t_l - tau - t_m
                    if diff > Delta_T:
                        bin_ser_low = bin_ser_mid + 1
                    elif diff < 0:
                        bin_ser_high = bin_ser_mid - 1
                    else:
                        r_precursor_i_j[i, j] += 1
                        break

            if numOfEvents_on_i:
                r_precursor_i_j[i, j] /= numOfEvents_on_i
            

            for m in np.arange(numOfEvents_on_j):
                t_m = grid_to_event_map[j, m]
                bin_ser_low = 0
                bin_ser_high = numOfEvents_on_i - 1
                while bin_ser_low <= bin_ser_high:
                    bin_ser_mid = math.floor( (bin_ser_low + bin_ser_high) / 2 )
                    t_l = grid_to_event_map[i, bin_ser_mid]
                    diff = t_l - tau - t_m
                    if diff > Delta_T:
                        bin_ser_high = bin_ser_mid - 1
                    elif diff < 0:
                        bin_ser_low = bin_ser_mid + 1
                    else:
                        r_trigger_i_j[i, j] += 1
                        break
            

            
            if numOfEvents_on_j:
                r_trigger_i_j[i, j] /= numOfEvents_on_j

@njit(nogil = True)
def binomial_term_new(n , r, a, b, lgdFctr ) :
    data_ = lgdFctr[ n ] - lgdFctr[ n - r ] - lgdFctr[ r ] + r * np.log(a) + ( n - r ) * np.log(b)
    return np.exp(data_)

@njit(parallel = True, nogil = True)
def find_p_value_new(number_of_grid_points, grdPnt_numOfEvent, Delta_T, tau, r_precursor_i_j, r_trigger_i_j, pval_precursor, pval_trigger, T, lgdFctr):
    prb_prec_coinc = Delta_T / (T - tau)
    for i in prange(number_of_grid_points):
        # if math.floor( i * 100 / number_of_grid_points ) != math.floor( ( i - 1 ) * 100 / number_of_grid_points ):
            # print(i*100/number_of_grid_points)
        for j in np.arange(number_of_grid_points):
            # if math.floor( j * 100 / number_of_grid_points ) != math.floor( ( j - 1 ) * 100 / number_of_grid_points ):
            #     print('\t',j*100/number_of_grid_points)
            numOfEvents_on_i = grdPnt_numOfEvent[i]
            numOfEvents_on_j = grdPnt_numOfEvent[j]
            
            prec_bnml_rgt_val = math.pow(1 - prb_prec_coinc, numOfEvents_on_j)
            prec_bnml_lft_val = 1 - prec_bnml_rgt_val
            r_precursor = r_precursor_i_j[i ,j] * numOfEvents_on_i
            for K_star in np.arange(round(r_precursor), numOfEvents_on_i + 1):
                pval_precursor[i, j] += binomial_term_new( numOfEvents_on_i, K_star, prec_bnml_lft_val, prec_bnml_rgt_val, lgdFctr )


            trig_bnml_rgt_val = math.pow(1 - prb_prec_coinc, numOfEvents_on_i)
            trig_bnml_lft_val = 1 - trig_bnml_rgt_val
            r_trigger = r_trigger_i_j[i ,j] * numOfEvents_on_j
            for K_star in np.arange(round(r_trigger), numOfEvents_on_j + 1):
                pval_trigger[i, j] += binomial_term_new( numOfEvents_on_j, K_star, trig_bnml_lft_val, trig_bnml_rgt_val, lgdFctr )


def createGPUFunction(sourceCode, functionName, block):
    from pycuda.compiler import SourceModule
    compilableSrc = sourceCode.replace("BLOCKDIM_X", str(block[0])).replace("BLOCKDIM_Y", str(block[1])).replace("BLOCKDIM_Z", str(block[2]))
    Mod = SourceModule(compilableSrc, no_extern_c = True)
    return Mod.get_function(functionName)
    
def printTime(time_spent):
    minutes_spent = math.floor(time_spent / 60)
    seconds_spent = round( time_spent - 60 * minutes_spent)
    print(f"Time elapsed to run the computation :: {minutes_spent} minutes {seconds_spent} seconds ")

class EventAnalysis:
    def __init__(self, rain_event_df, device_Id = None):
        #As defined in Event synchrony measures for functional climate network analysis: A case study on South American rainfall dynamics
        self.date_index = rain_event_df.index
        self.coordinate_columns = rain_event_df.columns
        self.rain_event_matrix = rain_event_df.to_numpy(copy = True)

        assert self.rain_event_matrix.dtype == np.bool, "The rain event dataframe does not have bool as its internal type, are you sure you sent an event series and not a time series?"
        # the rows are dates,and the columns are grid points.
        # the ex array will be all the index's where there is an event
        # (ex[0][i],ex[1][i]) => (grid index, date index) of event i
        self.index_of_events = np.where(self.rain_event_matrix.T)
        self.starting_date = self.date_index[0]
        self.ending_date = self.date_index[-1]
        self.T = ( self.ending_date - self.starting_date ).total_seconds() // number_of_seconds_per_hour
        # Number of events
        self.lx = self.index_of_events[1].size
        self.number_of_grid_points = self.rain_event_matrix.shape[1]
        self.ex = np.array(self.index_of_events, dtype="int32")

        self.grdPnt_numOfEvent_Map = np.asarray(np.unique(self.ex[0, :], return_counts=True))
        self.grid_point_with_maximun_events = self.grdPnt_numOfEvent_Map[0, np.argmax(self.grdPnt_numOfEvent_Map[1])]
        self.maximum_number_of_events = numOfEvents_onGridPoint(self.grdPnt_numOfEvent_Map, self.grid_point_with_maximun_events)
        self.grdPnt_numOfEvent = np.empty((self.number_of_grid_points), dtype = "int32")
        # creating and filling the grid to event map
        self.grid_to_event_map = np.zeros((self.number_of_grid_points, self.maximum_number_of_events), dtype="int32")
        for grid_point in range(self.number_of_grid_points):
            num_events = numOfEvents_onGridPoint(self.grdPnt_numOfEvent_Map, grid_point)
            time_series = self.date_index[self.ex[1,np.where(self.ex[0, :] == grid_point)][0]] - self.starting_date 
            self.grid_to_event_map[grid_point, 0:num_events] = time_series.total_seconds() // number_of_seconds_per_hour
            self.grdPnt_numOfEvent[grid_point] = num_events

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
        
        import pycuda.driver as cuda
        from pycuda.compiler import SourceModule
        cuda.init()
        self.ES_Source = open( os.path.join( os.path.dirname( __file__ ), "kernels", "event_Sync_.cu" ), "r" ).read()
        self.ECA_Source = open( os.path.join( os.path.dirname( __file__ ), "kernels", "event_Coinc_.cu" ), "r" ).read()
        self.ECA_Pval_Source = open( os.path.join( os.path.dirname( __file__ ), "kernels", "pValue_.cu" ), "r" ).read()

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



    def ES(self, tauMax = np.Inf):

        if self.lx == 0:  # Division by zero in output
            return np.nan, np.nan
        if self.lx in [1, 2]:  # Too few events to calculate
            return 0., 0.

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
        df = pd.DataFrame(Q, index = self.coordinate_columns, columns = self.coordinate_columns)
        df.index.name = "Coordinates"
        return df
    
    def ECA(self, Delta_T_obj, tau = 0, return_p_values = False, pValFillNA = True):
        if self.lx == 0:  # Division by zero in output
            return np.nan, np.nan
        if self.lx in [1, 2]:  # Too few events to calculate
            return 0., 0.

        r_precursor_i_j = np.zeros((self.number_of_grid_points, self.number_of_grid_points), dtype = np.float32)
        r_trigger_i_j = np.zeros((self.number_of_grid_points, self.number_of_grid_points), dtype = np.float32)

        Delta_T = Delta_T_obj.total_seconds() // number_of_seconds_per_hour
        
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

    def ECA_vec(self, Delta_T_objs, taus = 0, return_p_values = False, pValFillNA = True):
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
            Delta_T = Delta_T_objs[res].total_seconds() // number_of_seconds_per_hour
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

    def ES_Cuda(self, tauMax = np.Inf, block = None):
        import pycuda.driver as cuda
        from pycuda.compiler import SourceModule
        from pycuda._driver import device_attribute
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
        df = pd.DataFrame(Q, index = self.coordinate_columns, columns = self.coordinate_columns)
        df.index.name = "Coordinates"
        return df

    def ECA_Cuda(self, Delta_T_obj, tau = 0, return_p_values = False, block = None, pValFillNA = True):
        import pycuda.driver as cuda
        from pycuda.compiler import SourceModule
        from pycuda._driver import device_attribute
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

        Delta_T_GPU = np.array(Delta_T_obj.total_seconds() // number_of_seconds_per_hour, dtype=np.int32)
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

    def ECA_Cuda_vec(self, Delta_T_objs, taus = 0, return_p_values = False, block = None, pValFillNA = True):
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

        import pycuda.driver as cuda
        from pycuda.compiler import SourceModule
        from pycuda._driver import device_attribute
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

        Delta_T_GPU = np.array([Delta_T_obj.total_seconds() // number_of_seconds_per_hour for Delta_T_obj in Delta_T_objs], dtype=np.int32)
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
