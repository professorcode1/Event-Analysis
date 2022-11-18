#define i blockIdx.x
#define j blockIdx.y
#define tid_x threadIdx.x
#define tid_y threadIdx.y
#define tid_z threadIdx.y
#define bDm_x blockDim.x
#define bDm_y blockDim.y
#define bDm_z blockDim.z
#define WARP_SIZE 32

__forceinline__ __device__ int rowMajor_2D(int r, int c, int R, int C){
    return c + C * r;
}

__forceinline__ __device__ int colMajor_2D(int r, int c, int R, int C){
    return r + R * c;
}

extern "C"{
__global__ void ECA_kernel(float *r_precursor, float *r_trigger, const int * const grdPnt_numOfEvent_, const int * const grid_to_event_,  const int Delta_T, 
    const int Tau, const int num_of_grids, const int max_nm_events){
    __shared__ int  matrix_interm_buffer[ BLOCKDIM_X ];
    //matrix to temp store results (l/m) in a column major order
    __shared__ int outer_event_time_buffer[ BLOCKDIM_X ];
    __shared__ int inner_event_time_buffer[ BLOCKDIM_X ];
    __shared__ int s_i, s_j,  r_matrix_index;
    if( tid_x == 0 ){ 
        r_matrix_index = rowMajor_2D(i, j, num_of_grids, num_of_grids);
        r_precursor[ r_matrix_index ] = 0;
        r_trigger[ r_matrix_index ] = 0; 
        s_i = grdPnt_numOfEvent_[ i ];
        s_j = grdPnt_numOfEvent_[ j ];
    }
        
    __syncthreads();
    if (s_i < 1 || s_j < 1) return;
    int iter_i = 0, iter_j;
    
    while( iter_i * BLOCKDIM_X < s_i){
        
        int l = iter_i * BLOCKDIM_X + tid_x;
        
        if( l < s_i){
            outer_event_time_buffer[ tid_x ] = grid_to_event_[ i * max_nm_events + l];
        }
        
        matrix_interm_buffer[ tid_x ] = 0;

        // if( tid_x == 0 ){
            iter_j = 0;
        // } 

        // __syncthreads();

        while(iter_j * BLOCKDIM_X < s_j){
            
            int m = iter_j * BLOCKDIM_X + tid_x;
            
            if( m < s_j ){
                inner_event_time_buffer[ tid_x ] = grid_to_event_[ j * max_nm_events + m ];
            }

            int max_m = min( (iter_j + 1)  * BLOCKDIM_X, s_j) - ( iter_j * BLOCKDIM_X ) - 1;
            int bin_ser_low = 0, bin_ser_high = max_m;
            m = (bin_ser_low + bin_ser_high) / 2;
            
            __syncthreads();
            if( l < s_i ){
                for( int bin_ser_iter = 1 ; bin_ser_iter <= BLOCKDIM_X ; bin_ser_iter <<= 1){
                    int diff = outer_event_time_buffer[ tid_x ] - Tau - inner_event_time_buffer[ min( max(m, 0), max_m ) ];
                    char undershot = diff < 0 ;
                    char overshot =  diff > Delta_T ;
                    char perfect = (undershot | overshot) ^ 1 ;
                    bin_ser_low = ( overshot * ( m + 1 ) ) + ( ( overshot ^ 1 ) * bin_ser_low);
                    bin_ser_high = ( undershot * ( m - 1 ) ) + ( ( undershot ^ 1 ) * bin_ser_high );
                    m = (bin_ser_low + bin_ser_high) / 2;
                    matrix_interm_buffer[ tid_x ] |= perfect;
                }
            }
            __syncthreads();

            // if( tid_x == 0 ){
                iter_j ++;
            // }

        
        }

        for(int s=1 << (int)ceil(log2((float)BLOCKDIM_X) - 1) ; s > 0 ; s >>= 1){
            if(tid_x < s && tid_x + s < BLOCKDIM_X){
                matrix_interm_buffer[ tid_x ] += matrix_interm_buffer[ tid_x + s ];
            }
            __syncthreads();
        }

        if( tid_x == 0 ){
            r_precursor[ r_matrix_index ] += matrix_interm_buffer[ 0 ];
        }
            iter_i++;
        // __syncthreads();
    
    }

    if(tid_x == 0){
        r_precursor[ r_matrix_index ] /= s_i;
    }

    // __syncthreads();

        iter_i = 0;

    while( iter_i * BLOCKDIM_X < s_j ){
        
        int m = iter_i * BLOCKDIM_X + tid_x;
        
        if( m < s_j ){
            outer_event_time_buffer[ tid_x ] = grid_to_event_[ j * max_nm_events + m];
        }
        
        matrix_interm_buffer[ tid_x ] = 0;

        // if( tid_x == 0 ){
            iter_j = 0;
        // } 

        // __syncthreads();

        while(iter_j * BLOCKDIM_X < s_i){
            
            int l = iter_j * BLOCKDIM_X + tid_x;
            
            if( l < s_i ){
                inner_event_time_buffer[ tid_x ] = grid_to_event_[ i * max_nm_events + l ];
            }

            int max_l = min( (iter_j + 1)  * BLOCKDIM_X, s_i) - ( iter_j * BLOCKDIM_X ) - 1;
            int bin_ser_low = 0, bin_ser_high = max_l;
            l = (bin_ser_low + bin_ser_high) / 2;
            
            __syncthreads();
            if( m < s_j ){
                for( int bin_ser_iter = 1 ; bin_ser_iter <= BLOCKDIM_X ; bin_ser_iter <<= 1){
                    int diff = inner_event_time_buffer[ min( max(l, 0), max_l ) ] - Tau - outer_event_time_buffer[ tid_x ] ;
                    char undershot = diff < 0 ;
                    char overshot =  diff > Delta_T ;
                    char perfect = (undershot | overshot) ^ 1 ;
                    bin_ser_high = ( overshot * ( l - 1 ) ) + ( ( overshot ^ 1 ) * bin_ser_high);
                    bin_ser_low = ( undershot * ( l + 1 ) ) + ( ( undershot ^ 1 ) * bin_ser_low );
                    l = (bin_ser_low + bin_ser_high) / 2;
                    matrix_interm_buffer[ tid_x ] |= perfect;
                }
            }
            __syncthreads();

            // if( tid_x == 0 ){
                iter_j ++;
            // }

        
        }

        for(int s=1 << (int)ceil(log2((float)BLOCKDIM_X) - 1) ; s > 0 ; s >>= 1){
            if(tid_x < s && tid_x + s < BLOCKDIM_X){
                matrix_interm_buffer[ tid_x ] += matrix_interm_buffer[ tid_x + s ];
            }
            __syncthreads();
        }

        if( tid_x == 0 ){
            r_trigger[ r_matrix_index ] += matrix_interm_buffer[ 0 ];
        }
        // __syncthreads();
            iter_i++;
    
    }

    if(tid_x == 0){
        r_trigger[ r_matrix_index ] /= s_j;
    }

    

}
}