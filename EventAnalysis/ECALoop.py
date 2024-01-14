from numba import njit,jit, prange
import numpy as np

__all__ = [
    "ECA_Loop",
    "binomial_term_new",
    "find_p_value_new"
]

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
                    bin_ser_mid =(bin_ser_low + bin_ser_high) // 2 
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
                    bin_ser_mid = (bin_ser_low + bin_ser_high) // 2
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
def binomial_term_new(n:int, r:int, a:float|int, b:float|int, lgdFctr: np.ndarray) :
    data_ = lgdFctr[ n ] - lgdFctr[ n - r ] - lgdFctr[ r ] + r * np.log(a) + ( n - r ) * np.log(b)
    return np.exp(data_)

@njit(parallel = True, nogil = True)
def find_p_value_new(
    number_of_grid_points:int, grdPnt_numOfEvent:np.ndarray, 
    Delta_T:int, tau:int, 
    r_precursor_i_j:np.ndarray, r_trigger_i_j:np.ndarray, 
    pval_precursor:np.ndarray, pval_trigger:np.ndarray, 
    T:int, lgdFctr:np.ndarray
):
    prb_prec_coinc = Delta_T / (T - tau)
    for i in prange(number_of_grid_points):
        # if math.floor( i * 100 / number_of_grid_points ) != math.floor( ( i - 1 ) * 100 / number_of_grid_points ):
            # print(i*100/number_of_grid_points)
        for j in np.arange(number_of_grid_points):
            # if math.floor( j * 100 / number_of_grid_points ) != math.floor( ( j - 1 ) * 100 / number_of_grid_points ):
            #     print('\t',j*100/number_of_grid_points)
            numOfEvents_on_i = grdPnt_numOfEvent[i]
            numOfEvents_on_j = grdPnt_numOfEvent[j]
            
            prec_bnml_rgt_val = np.power(1 - prb_prec_coinc, numOfEvents_on_j)
            prec_bnml_lft_val = 1 - prec_bnml_rgt_val
            r_precursor = r_precursor_i_j[i ,j] * numOfEvents_on_i
            for K_star in np.arange(round(r_precursor), numOfEvents_on_i + 1):
                pval_precursor[i, j] += binomial_term_new( numOfEvents_on_i, K_star, prec_bnml_lft_val, prec_bnml_rgt_val, lgdFctr )


            trig_bnml_rgt_val = np.power(1 - prb_prec_coinc, numOfEvents_on_i)
            trig_bnml_lft_val = 1 - trig_bnml_rgt_val
            r_trigger = r_trigger_i_j[i ,j] * numOfEvents_on_j
            for K_star in np.arange(round(r_trigger), numOfEvents_on_j + 1):
                pval_trigger[i, j] += binomial_term_new( numOfEvents_on_j, K_star, trig_bnml_lft_val, trig_bnml_rgt_val, lgdFctr )
