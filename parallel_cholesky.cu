// parallel_cholesky_corrected.cu
// Compute triangular factors of an SPD matrix using GPU
// Improvized version: robust 1D/2D parallel Cholesky factorization
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <new>

#define MAX_MATRIX_SIZE 4096
#define TOL 1.0e-8

#define ERR_MALLOC 1
#define ERR_MEMCPY 2
#define ERR_KERNEL 3

// Define Matrix
typedef struct {
    int  n;             // order of matrix (number of rows and columns)
    double **elements;  // Allows access to array values as a matrix
    double *array;      // Linear array, stores matrix row-by-row
} Matrix;

// ----------------------- DEVICE KERNELS -----------------------------

// Initialize device matrix row pointers (parallelized)
__global__ void device_create_matrix_on_device(Matrix A) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < A.n) {
        A.elements[i] = &(A.array[i * A.n]);
    }
}

// Serial single-thread kernel (baseline)
__global__ void device_cholesky_factorization(Matrix A) {
    double sqrt_pivot;
    for (int k = 0; k < A.n; k++) {
        sqrt_pivot = sqrt(A.elements[k][k]);
        for (int j = k; j < A.n; j++) {
            A.elements[k][j] /= sqrt_pivot;
        }
        for (int i = k + 1; i < A.n; i++) {
            for (int j = k + 1; j < A.n; j++) {
                A.elements[i][j] -= A.elements[k][i] * A.elements[k][j];
            }
        }
        for (int j = k + 1; j < A.n; j++) {
            A.elements[j][k] = 0.0;
        }
    }
}

// ------------------- PARALLEL KERNELS -------------------------------

// Compute pivot = sqrt(A[k][k]) for row normalization
__global__ void device_cholesky_decompose_diagonal(Matrix A, double *pivots, int k) {
    if(threadIdx.x == 0 && blockIdx.x == 0){
        pivots[k] = sqrt(A.elements[k][k]);
    }
}

// Normalize row k using 1D parallelism
__global__ void device_cholesky_row_factorization(Matrix A, double *pivots, int k) {
    int j = k + blockIdx.x * blockDim.x + threadIdx.x;
    if(j < A.n){
        A.elements[k][j] /= pivots[k];
    }
}

// Update trailing submatrix and zero lower column
__global__ void device_cholesky_factorize_lower_matrix_block(Matrix A, int k) {
    int row = k + 1 + blockIdx.y * blockDim.y + threadIdx.y;
    int col = k + 1 + blockIdx.x * blockDim.x + threadIdx.x;

    // Update upper-triangular portion
    if(row < A.n && col < A.n && col >= row){
        A.elements[row][col] -= A.elements[k][row] * A.elements[k][col];
    }

    // Zero lower column k
    if(row < A.n && threadIdx.x == 0){
        A.elements[row][k] = 0.0;
    }
}

// -------------------- HOST UTILITY ROUTINES -------------------------

Matrix cholesky_factorization(Matrix&);   // host reference
Matrix product_with_transpose(Matrix& R);
int compare_matrix(Matrix&, Matrix&);
Matrix clone_matrix(Matrix& A);
void initialize_spd_matrix(Matrix&, double);
Matrix create_matrix(int, int);
void free_matrix_memory(Matrix& A);
void print_matrix(Matrix& A);
void check_error(cudaError_t, int);
void print_device_properties();
void print_usage(char* program_name);

// Host serial Cholesky factorization
Matrix cholesky_factorization(Matrix& A) {
    double sqrt_pivot;
    Matrix R = clone_matrix(A);
    for (int k = 0; k < R.n; k++) {
        sqrt_pivot = sqrt(R.elements[k][k]);
        for (int j = k; j < R.n; j++) {
            R.elements[k][j] /= sqrt_pivot;
        }
        for (int i = k + 1; i < R.n; i++) {
            for (int j = k + 1; j < R.n; j++) {
                R.elements[i][j] -= R.elements[k][i] * R.elements[k][j];
            }
        }
        for (int j = k + 1; j < R.n; j++) {
            R.elements[j][k] = 0.0;
        }
    }
    return R;
}

// Multiply R' * R
Matrix product_with_transpose(Matrix& R) {
    Matrix C = create_matrix(R.n, R.n);
    for (int i = 0; i < R.n; i++) {
        for (int j = 0; j < R.n; j++) {
            C.elements[i][j] = 0.0;
            for (int k = 0; k < R.n; k++)
                C.elements[i][j] += R.elements[k][i] * R.elements[k][j];
        }
    }
    return C;
}

// Compare two matrices within tolerance
int compare_matrix(Matrix& A, Matrix& B) {
    if (A.n != B.n) return 1;
    for (int i = 0; i < A.n; i++) {
        for (int j = 0; j < A.n; j++) {
            if (fabs(A.elements[i][j] - B.elements[i][j]) > TOL) return 1;
        }
    }
    return 0;
}

// Clone matrix
Matrix clone_matrix(Matrix& A) {
    Matrix C = create_matrix(A.n, A.n);
    for (int i = 0; i < A.n; i++)
        for (int j = 0; j < A.n; j++)
            C.elements[i][j] = A.elements[i][j];
    return C;
}

// Initialize SPD matrix
void initialize_spd_matrix(Matrix& A, double delta) {
    for (int i = 0; i < A.n; i++)
        A.elements[i][i] = delta;
    for (int i = 0; i < A.n; i++) {
        for (int j = i + 1; j < A.n; j++) {
            double value = (double)(rand()) / RAND_MAX;
            A.elements[i][j] = value;
            A.elements[j][i] = value;
            A.elements[i][i] += fabs(value);
            A.elements[j][j] += fabs(value);
        }
    }
}

// Create matrix
Matrix create_matrix(int n_rows, int n_cols) {
    Matrix A;
    A.n = n_rows;
    A.elements = new double*[A.n];
    A.array = new double[A.n * A.n];
    for (int i = 0; i < A.n; i++) A.elements[i] = &(A.array[i * A.n]);
    return A;
}

// Free matrix
void free_matrix_memory(Matrix& A) {
    if(A.elements) delete[] A.elements;
    if(A.array) delete[] A.array;
    A.elements = nullptr;
    A.array = nullptr;
}

// Print matrix
void print_matrix(Matrix& A) {
    for(int i=0;i<A.n;i++){
        for(int j=0;j<A.n;j++)
            printf("%8.4f ", A.elements[i][j]);
        printf("\n");
    }
}

// Check CUDA error
void check_error(cudaError_t err, int type) {
    if(err != cudaSuccess){
        switch(type){
            case ERR_MALLOC: fprintf(stderr,"Failed cudaMalloc: %s\n", cudaGetErrorString(err)); break;
            case ERR_MEMCPY: fprintf(stderr,"Failed cudaMemcpy: %s\n", cudaGetErrorString(err)); break;
            case ERR_KERNEL: fprintf(stderr,"Failed kernel launch: %s\n", cudaGetErrorString(err)); break;
        }
        exit(0);
    }
}

// Print device properties
void print_device_properties() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    cudaDeviceProp deviceProp;
    printf("------------------------------------------------------------\n");
    printf("Number of GPU devices found = %d\n", deviceCount);
    for(int i=0;i<deviceCount;i++){
        cudaGetDeviceProperties(&deviceProp,i);
        printf("[Device: %d] Compute Capability %d.%d\n",i,deviceProp.major,deviceProp.minor);
        printf(" ... multiprocessor count = %d\n",deviceProp.multiProcessorCount);
        printf(" ... max threads per multiprocessor = %d\n",deviceProp.maxThreadsPerMultiProcessor);
        printf(" ... max threads per block = %d\n",deviceProp.maxThreadsPerBlock);
        printf(" ... max block dim = %d,%d,%d\n",deviceProp.maxThreadsDim[0],deviceProp.maxThreadsDim[1],deviceProp.maxThreadsDim[2]);
        printf(" ... max grid size = %d,%d,%d\n",deviceProp.maxGridSize[0],deviceProp.maxGridSize[1],deviceProp.maxGridSize[2]);
        printf(" ... warp size = %d\n",deviceProp.warpSize);
        printf(" ... clock rate = %d MHz\n",deviceProp.clockRate/1000);
    }
    printf("------------------------------------------------------------\n");
}

// Print usage
void print_usage(char* program_name) {
    printf("Usage (serial): %s <matrix_size>\n",program_name);
    printf("Usage (parallel): %s <matrix_size> <block_size_1d> <block_size_x> <block_size_y>\n",program_name);
}

// --------------------------- MAIN PROGRAM -----------------------------
int main(int argc, char* argv[]){
    cudaError_t err = cudaSuccess;
    cudaEvent_t start, stop;
    float time_serial=0, time_parallel=0;

    print_device_properties();

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if(deviceCount>0) cudaSetDevice(0);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    if(argc!=2 && argc!=5){ print_usage(argv[0]); exit(0); }

    int N = atoi(argv[1]);
    if(N>MAX_MATRIX_SIZE){ printf("Max matrix size: %d\n",MAX_MATRIX_SIZE); exit(0); }

    int block_size_1d=1024, block_size_x=32, block_size_y=32;
    bool run_parallel=false;
    if(argc==5){
        run_parallel=true;
        block_size_1d=atoi(argv[2]);
        block_size_x=atoi(argv[3]);
        block_size_y=atoi(argv[4]);
    }

    printf("Matrix size: %d\n",N);
    if(run_parallel) printf("1D block %d, 2D block %d x %d\n",block_size_1d, block_size_x, block_size_y);
    else printf("Running serial baseline only.\n");

    Matrix A = create_matrix(N,N);
    initialize_spd_matrix(A,1.0);
    Matrix A_copy = clone_matrix(A);

    // -------- SERIAL IMPLEMENTATION -----------
    Matrix dA_serial;
    dA_serial.n = N;
    size_t size_elements = N*sizeof(double*);
    size_t size_array = N*N*sizeof(double);
    cudaMalloc(&dA_serial.elements,size_elements); check_error(err,ERR_MALLOC);
    cudaMalloc(&dA_serial.array,size_array); check_error(err,ERR_MALLOC);
    cudaMemcpy(dA_serial.array,A.array,size_array,cudaMemcpyHostToDevice); check_error(err,ERR_MEMCPY);

    int threads = 256;
    int blocks = (N+threads-1)/threads;
    device_create_matrix_on_device<<<blocks,threads>>>(dA_serial);
    err = cudaGetLastError(); check_error(err,ERR_KERNEL);

    cudaEventRecord(start,0);
    device_cholesky_factorization<<<1,1>>>(dA_serial);
    err = cudaGetLastError(); check_error(err,ERR_KERNEL);
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_serial,start,stop);

    Matrix R_serial = create_matrix(N,N);
    cudaMemcpy(R_serial.array,dA_serial.array,size_array,cudaMemcpyDeviceToHost); check_error(err,ERR_MEMCPY);

    Matrix RtR_serial = product_with_transpose(R_serial);
    int err_serial = compare_matrix(A,RtR_serial);
    if(err_serial!=0) printf("+++ Serial factorization failed!\n");
    else printf("+++ Serial factorization OK, time %.4f ms\n",time_serial);

    // -------- PARALLEL IMPLEMENTATION -----------
    if(run_parallel){
        Matrix dA_parallel; dA_parallel.n=N;
        cudaMalloc(&dA_parallel.elements,size_elements); check_error(err,ERR_MALLOC);
        cudaMalloc(&dA_parallel.array,size_array); check_error(err,ERR_MALLOC);
        cudaMemcpy(dA_parallel.array,A_copy.array,size_array,cudaMemcpyHostToDevice); check_error(err,ERR_MEMCPY);

        device_create_matrix_on_device<<<blocks,threads>>>(dA_parallel);
        err = cudaGetLastError(); check_error(err,ERR_KERNEL);

        double *d_pivots=nullptr;
        cudaMalloc(&d_pivots,N*sizeof(double)); check_error(err,ERR_MALLOC);

        cudaEventRecord(start,0);
        for(int k=0;k<N;k++){
            device_cholesky_decompose_diagonal<<<1,1>>>(dA_parallel,d_pivots,k);
            err=cudaGetLastError(); check_error(err,ERR_KERNEL);

            int cols_to_process = N - k;
            int gridDim_row = (cols_to_process+block_size_1d-1)/block_size_1d;
            device_cholesky_row_factorization<<<gridDim_row,block_size_1d>>>(dA_parallel,d_pivots,k);
            err=cudaGetLastError(); check_error(err,ERR_KERNEL);

            int trailing = N-(k+1);
            if(trailing>0){
                dim3 block2D(block_size_x, block_size_y);
                dim3 grid2D((trailing+block2D.x-1)/block2D.x,(trailing+block2D.y-1)/block2D.y);
                device_cholesky_factorize_lower_matrix_block<<<grid2D,block2D>>>(dA_parallel,k);
                err=cudaGetLastError(); check_error(err,ERR_KERNEL);
            }
        }
        cudaEventRecord(stop,0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time_parallel,start,stop);

        Matrix R_parallel = create_matrix(N,N);
        cudaMemcpy(R_parallel.array,dA_parallel.array,size_array,cudaMemcpyDeviceToHost); check_error(err,ERR_MEMCPY);

        Matrix RtR_parallel = product_with_transpose(R_parallel);
        int err_parallel = compare_matrix(A_copy,RtR_parallel);
        if(err_parallel!=0) printf("+++ Parallel factorization failed!\n");
        else printf("+++ Parallel factorization OK, time %.4f ms\n",time_parallel);

        int cmp = compare_matrix(R_serial,R_parallel);
        if(cmp==0) printf("+++ Serial and parallel R match!\n");
        else printf("+++ Serial and parallel R differ!\n");

        printf("Speedup: %.2fx\n",time_serial/time_parallel);

        cudaFree(d_pivots);
        cudaFree(dA_parallel.elements);
        cudaFree(dA_parallel.array);
        free_matrix_memory(R_parallel);
        free_matrix_memory(RtR_parallel);
        free_matrix_memory(A_copy);
    }

    cudaFree(dA_serial.elements);
    cudaFree(dA_serial.array);
    free_matrix_memory(A);
    free_matrix_memory(R_serial);
    free_matrix_memory(RtR_serial);

    return 0;
}
