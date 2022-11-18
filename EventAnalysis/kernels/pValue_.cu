  #define i blockIdx.x
  #define j blockIdx.y
  #define tid_x threadIdx.x
  #define tid_y threadIdx.y
  #define tid_z threadIdx.y
  #define bDm_x blockDim.x
  #define bDm_y blockDim.y
  #define bDm_z blockDim.z
  #define WARP_SIZE 32
  #define CUDART_PI_F 3.14159265f
  __forceinline__ __device__ int rowMajor_2D(int r, int c, int R, int C){
      return c + C * r;
  }

  __forceinline__ __device__ int colMajor_2D(int r, int c, int R, int C){
      return r + R * c;
  }

  extern "C"{
  __global__ void pVal_kernel( double *pVal_precursor, double *pVal_trigger, float const * const r_precursor, float const * const r_trigger, 
      const int * const grdPnt_numOfEvent_,  const int Delta_T,  const int Tau, const int num_of_grids, const int T, const double * const logedFactorial){
          
      __shared__ double answer_collection_buffer[ BLOCKDIM_X ];
      __shared__ double prb_prec_coinc_shared;
      __shared__ double prec_bnml_rgt_lg_shared;
      __shared__ double prec_bnml_lft_lg_shared;

      if( tid_x == 0 ){
          prb_prec_coinc_shared = ((double)Delta_T / (double)(T - Tau));
          prec_bnml_rgt_lg_shared = pow(1 - prb_prec_coinc_shared, grdPnt_numOfEvent_[ j ] ) ; 
          prec_bnml_lft_lg_shared = log(1 - prec_bnml_rgt_lg_shared) ;
          prec_bnml_rgt_lg_shared = log(prec_bnml_rgt_lg_shared);
          pVal_precursor[ rowMajor_2D(i, j, num_of_grids, num_of_grids) ] = 0;
      }
      __syncthreads();
      double prec_bnml_rgt_val = prec_bnml_rgt_lg_shared;
      double prec_bnml_lft_val = prec_bnml_lft_lg_shared;
      
      int K_end = grdPnt_numOfEvent_[ i ];
      int K_start = round( r_precursor[rowMajor_2D(i, j, num_of_grids, num_of_grids)] * K_end );
      int iter_i = 0;
      while( K_start + iter_i * BLOCKDIM_X < K_end ){
          answer_collection_buffer[ tid_x ] = 0;
          int K = K_start + iter_i * BLOCKDIM_X + tid_x;
          if( K <= K_end ){
              answer_collection_buffer[ tid_x ] = exp( logedFactorial[ K_end ] - logedFactorial[ K ] - logedFactorial[ K_end - K ] + K * prec_bnml_lft_val + ( K_end - K ) * prec_bnml_rgt_val );
          }

          __syncthreads();
          for(int s=1 << (int)ceil(log2((float)BLOCKDIM_X) - 1) ; s > 0 ; s >>= 1){
              if(tid_x < s && tid_x + s < BLOCKDIM_X){
                  answer_collection_buffer[ tid_x ] += answer_collection_buffer[ tid_x + s ];
              }
              __syncthreads();
          }
          if( tid_x == 0 ){
              pVal_precursor[ rowMajor_2D(i, j, num_of_grids, num_of_grids) ] += answer_collection_buffer[0];
          }
          iter_i++;
      }

      __syncthreads();

      if( tid_x == 0 ){
          prec_bnml_rgt_lg_shared = pow( 1 - prb_prec_coinc_shared, grdPnt_numOfEvent_[ i ] ) ; 
          prec_bnml_lft_lg_shared = log( 1 - prec_bnml_rgt_lg_shared ) ;
          prec_bnml_rgt_lg_shared = log( prec_bnml_rgt_lg_shared );
          pVal_trigger[ rowMajor_2D(i, j, num_of_grids, num_of_grids) ] = 0;
      }
      __syncthreads();
      prec_bnml_rgt_val = prec_bnml_rgt_lg_shared;
      prec_bnml_lft_val = prec_bnml_lft_lg_shared;
      
      K_end = grdPnt_numOfEvent_[ j ];
      K_start = round( r_trigger[rowMajor_2D(i, j, num_of_grids, num_of_grids)] * K_end );
      iter_i = 0;
      while( K_start + iter_i * BLOCKDIM_X < K_end ){
          answer_collection_buffer[ tid_x ] = 0;
          int K = K_start + iter_i * BLOCKDIM_X + tid_x;
          if( K <= K_end ){
              answer_collection_buffer[ tid_x ] = exp( logedFactorial[ K_end ] - logedFactorial[ K ] - logedFactorial[ K_end - K ] + K * prec_bnml_lft_val + ( K_end - K ) * prec_bnml_rgt_val );
          }

          __syncthreads();
          for(int s=1 << (int)ceil(log2((float)BLOCKDIM_X) - 1) ; s > 0 ; s >>= 1){
              if(tid_x < s && tid_x + s < BLOCKDIM_X){
                  answer_collection_buffer[ tid_x ] += answer_collection_buffer[ tid_x + s ];
              }
              __syncthreads();
          }
          if( tid_x == 0 ){
              pVal_trigger[ rowMajor_2D(i, j, num_of_grids, num_of_grids) ] += answer_collection_buffer[0];
          }
          iter_i++;
      }


  }
  }