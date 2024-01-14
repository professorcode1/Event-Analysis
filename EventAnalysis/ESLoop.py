from numba import njit,jit, prange
import numpy as np

__all__ = [
    "ES_Loop",
    "normaliseQMatrix"
]

@njit(parallel = True, nogil = True)
def normaliseQMatrix(Q, grdPnt_numOfEvent):
    (number_of_grid_points, _) = Q.shape
    for i in prange(number_of_grid_points):
        number_of_events_i = grdPnt_numOfEvent[ i ]
        if number_of_events_i <= 2:
            continue
        Q[i, i] /= ( number_of_events_i - 2 )
        for j in np.arange( i ):
            number_of_events_j = grdPnt_numOfEvent[ j ]
            if number_of_events_j <= 2:
                continue
            normal_factor = np.sqrt( ( number_of_events_i - 2 ) * ( number_of_events_j - 2 ) )
            Q[i, j] /= normal_factor
            Q[j, i] /= normal_factor

@njit(parallel = True, nogil = True)
def ES_Loop(
    number_of_grid_points:int, grdPnt_numOfEvent:np.ndarray, 
    Tau_hlpr:np.ndarray, tauMax:float, 
    grid_to_event_map:np.ndarray, c_i_j:np.ndarray
):
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