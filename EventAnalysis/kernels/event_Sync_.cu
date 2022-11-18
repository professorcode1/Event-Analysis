#define i blockIdx.x
#define j blockIdx.y
#define tid_x threadIdx.x
#define tid_y threadIdx.y
#define bDm_x blockDim.x
#define bDm_y blockDim.y
#define WARP_SIZE 32

__forceinline__ __device__ int rowMajor_2D(int r, int c, int R, int C){
    return c + C * r;
}

extern "C"{
__global__ void ES_kernel(float * C, int const * const grdPnt_numOfEvent_,int const * const grid_to_event_, const int num_of_grids, const int max_nm_events){
    // int i = blockIdx.x;
    // int j = blockIdx.y;
    // int tid_x = threadIdx.x;
	// int tid_y = threadIdx.y;
    // blockIdx, threadIdx, blockDim are stored in a special register, copying it into a variable will waste time and space
    __shared__ float j_per_conv[ BLOCKDIM_X * BLOCKDIM_Y ];
    __shared__ int grid_to_event_i[ BLOCKDIM_X + 3 ];
    __shared__ int grid_to_event_j[ BLOCKDIM_Y + 3 ];
    __shared__ int  s_i, s_j, total_threads, max_nm_events_cached, C_matrix_index;
    
    
    int thread_id = rowMajor_2D(tid_x, tid_y, bDm_x, bDm_y);
    if(thread_id == 0){
        C_matrix_index = rowMajor_2D(i,j,num_of_grids,num_of_grids);
        C[C_matrix_index] = 0;
        s_i = grdPnt_numOfEvent_[ i ];
        s_j = grdPnt_numOfEvent_[ j ];
        total_threads = bDm_x * bDm_y;
        max_nm_events_cached = max_nm_events;
    }
    __syncthreads();
    
    if (s_i < 3 || s_j < 3) return;
    int iter_i = 0, iter_j;
    while( bDm_x * iter_i <  s_i - 2){
        
        int l = (bDm_x * iter_i + tid_x) + 1;

        if(tid_y == 0 && l <= s_i - 2)
            grid_to_event_i[tid_x + 2] = grid_to_event_[i * max_nm_events_cached +  l];
        
        if( thread_id == 0 ){
            grid_to_event_i[ 1 ] = grid_to_event_[ i * max_nm_events_cached +  bDm_x * iter_i ];
            if(iter_i){
                grid_to_event_i[0] = grid_to_event_[ i * max_nm_events_cached +  bDm_x * iter_i - 1];
            }else{
                grid_to_event_i[0] = -1;
            }
            int max_l = min(bDm_x * (iter_i + 1) + 1, s_i - 1);
            int end_event_index = max_l - bDm_x * iter_i + 1;
            grid_to_event_i[ end_event_index ] = grid_to_event_[ i * max_nm_events_cached + max_l ];
        }
        // __syncthreads();
        
            iter_j = 0;
        while( bDm_y * iter_j < s_j - 2 ){
            
            j_per_conv[ thread_id ] = 0;
            
            int m = (bDm_y * iter_j + tid_y) + 1;

            if(l <= s_i - 2 && m <= s_j - 2){
                if(tid_x == 0)
                    grid_to_event_j[tid_y + 1] = grid_to_event_[j * max_nm_events_cached +  m];

                if( thread_id == 0 ){
                    grid_to_event_j[ 0 ] = grid_to_event_[ j * max_nm_events_cached +   bDm_y * iter_j ];
                    if(s_j - 1 > bDm_y * (iter_j + 1) + 1){
                        int max_m = bDm_y * (iter_j + 1) + 1;
                        int end_event_index = bDm_y + 1 ;
                        grid_to_event_j[ end_event_index     ] = grid_to_event_[ j * max_nm_events_cached + max_m ];
                        grid_to_event_j[ end_event_index + 1 ] = grid_to_event_[ j * max_nm_events_cached + max_m + 1];
                    }else{
                        int max_m = s_j - 1;
                        int end_event_index = max_m - bDm_y * iter_j;
                        grid_to_event_j[ end_event_index     ] = grid_to_event_[ j * max_nm_events_cached + max_m ];
                        grid_to_event_j[ end_event_index + 1 ] = -1; 
                    }
                }
            }
            __syncthreads();

            if(l <= s_i - 2 && m <= s_j - 2){
                char J_is_one, J_is_half;{
                    int t_l_minus_two = grid_to_event_i[tid_x];
                    int t_l_minus_one = grid_to_event_i[tid_x + 1];
                    int t_l = grid_to_event_i[tid_x + 2];
                    int t_l_plus_one = grid_to_event_i[tid_x + 3];
                    int t_m_minus_one = grid_to_event_j[tid_y];
                    int t_m = grid_to_event_j[tid_y + 1];
                    int t_m_plus_one = grid_to_event_j[tid_y + 2];
                    int t_m_plus_two = grid_to_event_j[tid_y + 3];
    
                    char sigma_i_j_l_m, sigma_j_i_m_l_minus_one , sigma_j_i_m_plus_one_l;{
                        int two_tao = min(min(t_l_plus_one - t_l, t_l - t_l_minus_one) , min(t_m_plus_one - t_m , t_m - t_m_minus_one));
                        int time_dif = t_l - t_m;
                        sigma_i_j_l_m = (0 < time_dif) & ((2 * time_dif) <= two_tao);
                    }{
                        
                        int two_tao;
                        if (t_l_minus_two +1)
                            two_tao = min(min(t_l - t_l_minus_one, t_l_minus_one - t_l_minus_two), min(t_m_plus_one - t_m , t_m - t_m_minus_one));
                        else
                            two_tao = min(t_l - t_l_minus_one, min(t_m_plus_one - t_m , t_m - t_m_minus_one));
                        int time_dif = t_m - t_l_minus_one;
                        sigma_j_i_m_l_minus_one = (0 < time_dif) & ((2 * time_dif) <= two_tao);
                    }{
                        int two_tao;
                        if(t_m_plus_two + 1)
                            two_tao = min(min(t_l_plus_one - t_l, t_l - t_l_minus_one) , min(t_m_plus_two - t_m_plus_one, t_m_plus_one - t_m));
                        else
                            two_tao = min(min(t_l_plus_one - t_l, t_l - t_l_minus_one) , t_m_plus_one - t_m);
                        int time_dif = t_m_plus_one - t_l;
                        sigma_j_i_m_plus_one_l = (0 < time_dif) & ((2 * time_dif) <= two_tao);
                    }
                    char intermediate_condition = sigma_j_i_m_l_minus_one | sigma_j_i_m_plus_one_l;
                    J_is_one = sigma_i_j_l_m &  (!intermediate_condition);
                    J_is_half = (t_l == t_m) | (sigma_i_j_l_m & intermediate_condition);
                }
                j_per_conv[thread_id] += 0.5f * J_is_half;  
                j_per_conv[thread_id] += J_is_one;
            }
            __syncthreads();

            // for(int s = 1 << (int) ceil(log2((double)total_threads) - 1); s > WARP_SIZE ; s >>= 1){
            //     if(thread_id < s && thread_id + s < total_threads)
            //         j_per_conv[thread_id] += j_per_conv[thread_id + s];
                
            //     __syncthreads();
            // }
            // if(thread_id < WARP_SIZE)
            //     warpReduce(j_per_conv, thread_id);

            for(int s = 1 << (int) ceil(log2((double)total_threads) - 1); s > 0 ; s >>= 1){
                if(thread_id < s && thread_id + s < total_threads)
                    j_per_conv[thread_id] += j_per_conv[thread_id + s];
                
                __syncthreads();
            }


            if(thread_id == 0){
                C[C_matrix_index] += j_per_conv[0];
            }
            // __syncthreads();
                iter_j++;
        }

        // if(thread_id == 0)
            iter_i++;

        // __syncthreads();

    }
}
}