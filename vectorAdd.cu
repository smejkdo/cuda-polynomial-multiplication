/**
 * @file polynomial.cu
 * @author Dominik Šmejkal
 * @brief polynomial multiplication on gpu
 * @version 0.1
 * @date 2022-03-01
 * 
 * @copyright Copyright (c) 2022
 * 
 */

//CPU - R5 3600 12T
//GPU - RTX 2070S 2560CUDA
#include <iostream>
#include <chrono>
using namespace std;

#define GPUTHREADS 2560
#define CPUTHREADS 12
#define COEFDIFF (long long int)889516852 //rozdil mezi int max a sqrt long long max
//#define SIZE 2048*64
#define GPU_MAX_THREADS 1024
#define KARATSUBA_EXPANSION_BOUND SIZE-1
#define COEFTYPE int//long long int

int SIZE = 2048*256;
void printPolynomial(COEFTYPE* p, unsigned int size);

COEFTYPE* emptyInitialization(unsigned int size){
    COEFTYPE * polynomial = (COEFTYPE *)malloc(size * sizeof(COEFTYPE));
    if (polynomial == NULL) {
     fprintf(stderr, "Failed to allocate host vector!\n");
     exit(EXIT_FAILURE);
    }
    return polynomial;
}

/**
 * @brief Initializes empty polynomial, either by zeros or by random numbers
 * 
 * @param size degree+1 of polynomial
 * @param zeros true if initialization by zeros
 * @return COEFTYPE* 
 */
COEFTYPE* initPolynomial(unsigned int size, bool zeros=false){
    COEFTYPE * polynomial = emptyInitialization(size);
    if (zeros)
        for (unsigned int i = 0; i < size; i++)
            polynomial[i] = 0; 
    else
        for (unsigned int i = 0; i < size; i++)
            polynomial[i] = rand()%10; //(rand() + COEFDIFF);
    return polynomial;
}

//CPU PART  //CPU PART  //CPU PART  //CPU PART  //CPU PART  //CPU PART  //CPU PART  //CPU PART  //CPU PART  //CPU PART  //CPU PART  //CPU PART  //CPU PART
//CPU PART  //CPU PART  //CPU PART  //CPU PART  //CPU PART  //CPU PART  //CPU PART  //CPU PART  //CPU PART  //CPU PART  //CPU PART  //CPU PART  //CPU PART
//CPU PART  //CPU PART  //CPU PART  //CPU PART  //CPU PART  //CPU PART  //CPU PART  //CPU PART  //CPU PART  //CPU PART  //CPU PART  //CPU PART  //CPU PART


/**
 * @brief Calculates a coeficient for a given rank.
 * 
 * @param rank rank of calculated coeficient
 * @param p_A first polynomial
 * @param p_B second polynomial
 * @param lower first lower index (inclusive)
 * @param upper last upper index (exclusive)
 * @return COEFTYPE 
 */
COEFTYPE multiplicationCoeficient(size_t rank, COEFTYPE* p_A, COEFTYPE* p_B, size_t lower, size_t upper){
    size_t size = upper - lower;
    COEFTYPE coef = 0;
    for (size_t j = 0; j < rank+1; j++){
        if (j >= size || rank-j >= size)
            continue;
        coef += p_A[j+lower]*p_B[rank-j+lower];
    }
    return coef;
}

/**
 * @brief Calculates a multiplication of polynomials by calculating each coeficient.
 * 
 * @param p_A first polynomial
 * @param p_B second polynomial
 * @param lower first lower index (inclusive)
 * @param upper last upper index (exclusive)
 * @return COEFTYPE* Returns new polynomial
 */
COEFTYPE* simpleMultiplyPolynomials(COEFTYPE* p_A, COEFTYPE* p_B, size_t lower, size_t upper){
    size_t size = upper - lower;
    COEFTYPE* p_C = emptyInitialization(2 * size);
    for (size_t i = 0; i < 2 * size; i++){
        p_C[i] = multiplicationCoeficient(i,p_A,p_B, lower, upper);

    }
    
    return p_C;
}

/**
 * @brief Calculates a multiplication of polynomials using a formula c[i+j]=a[i]*b[j]
 * 
 * @param p_A first polynomial
 * @param p_B second polynomial
 * @param lower first lower index (inclusive)
 * @param upper last upper index (exclusive)
 * @return COEFTYPE* Returns new polynomial
 */
COEFTYPE* flattenedMultiplyPolynomials(COEFTYPE* p_A, COEFTYPE* p_B, size_t lower, size_t upper){
    size_t size = upper - lower;
    COEFTYPE* p_C = initPolynomial(2 * size, true);
        #pragma omp parallel for \
	   	default(shared) \
	   	schedule(dynamic)
        for (size_t i = 0; i < size; i++){
            for (size_t j = 0; j < size; j++)
                p_C[i+j] += p_A[i+lower]*p_B[j+lower];
        }

    
    return p_C;
} 


/**
 * @brief Sums upper and lower half of polynomial for use in karatsuba algorithm
 * 
 * @param p_A polynomial
 * @param lower first lower index (inclusive)
 * @param half first index of upper half
 * @param upper last upper index (exclusive)
 * @return COEFTYPE* returns a polynomial of half size
 */
COEFTYPE* sumHalves(COEFTYPE* p_A, size_t lower, size_t half, size_t upper){
    COEFTYPE* h; 
    if (2*half < upper){
        h = emptyInitialization(half-lower+1);
        h[half-lower] = p_A[upper-1];
    }
    else
        h = emptyInitialization(half-lower);
    for (size_t i = 0; i < half; i++){
        h[i] = p_A[i+lower] + p_A[i+half];
    }
    return h;
}

/**
 * @brief Subtracts coeficients of right polynomial from coeficients of left polynomial
 * 
 * @param left left polynomial
 * @param right right polynomial
 * @param leftSize size of left polynomial
 * @param rightSize size of right polynomial
 * @return COEFTYPE* returns polynomial
 */
COEFTYPE* polynomialSubtract(COEFTYPE* left, COEFTYPE* right, size_t leftSize, size_t rightSize){
    
    size_t size, bound;
    if (leftSize < rightSize){
        size = rightSize;
        bound = leftSize;
        }
    else{
        size = leftSize;
        bound = rightSize;
    }
    COEFTYPE* result = emptyInitialization(size);
    for (size_t i = 0; i < bound; i++)
        result[i] = left[i] - right[i];

    if (leftSize < rightSize)
        for (size_t i = bound; i < size; i++)
            result[i] = -right[i];
    else if (leftSize > rightSize)
        for (size_t i = bound; i < size; i++)
            result[i] = left[i];
    return result;
}

/**
 * @brief Connects two polynomials into one (like vectors)
 * 
 * @param base first polynomial
 * @param attached second polynomial
 * @param offset index from which the second polynomial is attached
 * @param baseSize size of first polynomial
 * @param attachedSize size of second polynomial
 * @return COEFTYPE* returns polynomial
 */
COEFTYPE* concat(COEFTYPE* base, COEFTYPE* attached, size_t offset, size_t baseSize, size_t attachedSize){
    COEFTYPE* result;
    result = emptyInitialization(offset + attachedSize);
    for (size_t i = 0; i < baseSize; i++)
        result[i] = base[i];
    for (size_t i = baseSize; i < offset + attachedSize; i++)
        result[i] = 0;
    for (size_t i = offset; i < offset + attachedSize; i++)
        result[i] += attached[i-offset];

    return result;
}

COEFTYPE* GPUKaratsubaChoice(COEFTYPE* p_A, 
                            COEFTYPE* p_B, 
                            size_t lower, 
                            size_t upper);

COEFTYPE* karatsubaMultiplicationChoice(COEFTYPE* p_A, 
                                        COEFTYPE* p_B, 
                                        size_t lower, 
                                        size_t upper, 
                                        bool onGpu){
    if(onGpu)
        return GPUKaratsubaChoice(p_A,p_B, lower, upper);
    else
        return flattenedMultiplyPolynomials(p_A,p_B, lower, upper);
}

/**
 * @brief Computes polynomial multiplication by using karatsuba algorithm
 * ( (E1*D1) * (x^n) + ((E1 + E0) * (D0 + D1) - (D1*E1) - (D0*E0)) * (x ^ n/2) + E0D0)
 * 
 * @param p_A first polynomial
 * @param p_B second polynomial
 * @param lower first lower index (inclusive)
 * @param upper last upper index (exclusive)
 * @return COEFTYPE* returns polynomial
 */
COEFTYPE* karatsubaMultiplyPolynomials(COEFTYPE* p_A, 
                                        COEFTYPE* p_B, 
                                        size_t lower, 
                                        size_t upper, 
                                        bool onGpu){
    size_t size = upper - lower;
    COEFTYPE *l,*u,*x, *result, *sum_A, *sum_B, *x_sub_A, *x_sub_AB, *l_x;
    if (upper - lower <= KARATSUBA_EXPANSION_BOUND)
        return karatsubaMultiplicationChoice(p_A,p_B,lower,upper,onGpu);

    size_t half = (upper + lower)/2;
    size_t halfSize = upper - half;
    l = karatsubaMultiplyPolynomials(p_A,p_B, lower, half, onGpu);
    u = karatsubaMultiplyPolynomials(p_A,p_B, half, upper, onGpu);
    sum_A = sumHalves(p_A, lower, half, upper);
    sum_B = sumHalves(p_B, lower, half, upper);

    x = karatsubaMultiplyPolynomials(sum_A, sum_B, 0, halfSize, onGpu);
    x_sub_A = polynomialSubtract(x, l, size, size);
    x_sub_AB = polynomialSubtract(x_sub_A, u, size, size);
    l_x = concat(l, x_sub_AB, halfSize, size, size);
    result = concat(l_x, u, size, 1.5*size, size);

    /*if(onGpu){
        cudaFree(l);
        cudaFree(u);
        cudaFree(x);
        cudaFree(l_x);
        cudaFree(sum_A);
        cudaFree(sum_B);
        cudaFree(x_sub_A);
        cudaFree(x_sub_AB);
    }*/
        free(l);
        free(u);
        free(x);
        free(l_x);
        free(sum_A);
        free(sum_B);
        free(x_sub_A);
        free(x_sub_AB);
    
    return result;
}
/**
    cudaMalloc( (void**)&d_A, s_upper * sizeof(COEFTYPE) );
    cudaFree(d_A);
 */

/**
 * @brief Multiplies polynomials by algorithm chosen by choice
 * 
 * @param p_A first polynomial
 * @param p_B second polynomial
 * @param choice choice of algorithm 0-2
 * @return COEFTYPE* returns polynomial
 */
COEFTYPE* multiplyPolynomials(COEFTYPE* p_A, COEFTYPE* p_B, int choice = 0){
    switch (choice)
    {
    case 1:
        return flattenedMultiplyPolynomials(p_A,p_B, 0, SIZE);
    case 2:
        return karatsubaMultiplyPolynomials(p_A,p_B, 0, SIZE, false);
    default:
        return simpleMultiplyPolynomials(p_A,p_B, 0, SIZE);
    }
}

/**
 * @brief Prints polynomial to stdout
 * 
 * @param p polynomial
 * @param size size of polynomial p
 */
void printPolynomial(COEFTYPE* p, unsigned int size){
    for (unsigned int i = 0; i < size; i++){
        cout << p[i] << ", ";
    }
    cout << "\n";

}

double epsilon = 0.01;
bool approxEqual(COEFTYPE a, COEFTYPE b){
    return abs(a-b) > epsilon;
}

/**
 * @brief Verifies if the given results are the same, prints the 
 * number of errors on stdout.Expects polynomials of size SIZE*2.
 * 
 * @param p_A first polynmial
 * @param p_B second polynomial
 * @param p_C third polynomial
 * @return true results are returning the same values
 * @return false results are returning different values
 */
bool verifyResults(COEFTYPE* p_A, COEFTYPE* p_B, COEFTYPE* p_C){
    int errors = 0;
    for (size_t i = 0; i < SIZE*2; i++)
        if(approxEqual(p_A[i], p_B[i]) || approxEqual(p_B[i],p_C[i]))
        //if ((int)p_A[i] != (int)p_B[i] || (int)p_B[i] != (int)p_C[i])
        //cout << p_A[i] << " " << p_B[i] << " " << p_C[i] << "\n";
            errors++;
    cout << "Errors: " << errors << "\n";
    return !errors;
}

/**
 * @brief Runs all 3 algorithms for multiplication on cpu and prints to stdout time of their run in nanoseconds
 * 
 * @param p_A first polynomial
 * @param p_B second polynomial
 */
void runOnCPU(COEFTYPE* p_A, COEFTYPE* p_B){
    COEFTYPE *p_C1,*p_C2,*p_C3;
    auto start = chrono::steady_clock::now();
    p_C1 = multiplyPolynomials(p_A,p_B);
    auto end = chrono::steady_clock::now();
    double simpleTime = chrono::duration_cast<chrono::nanoseconds>(end - start).count()/1e9;
    cout << "Simple: " << simpleTime << " s\n";
    //printPolynomial(p_C1, 2*SIZE-1);
    
    start = chrono::steady_clock::now();
    p_C2 = multiplyPolynomials(p_A,p_B,1);
    end = chrono::steady_clock::now();
    double flattenedTime = chrono::duration_cast<chrono::nanoseconds>(end - start).count()/1e9;
    cout << "Flattened: " << flattenedTime << " s\n";
    //printPolynomial(p_C2, 2*SIZE-1);

    start = chrono::steady_clock::now();
    p_C3 = multiplyPolynomials(p_A,p_B,2);
    end = chrono::steady_clock::now();
    double karatsubaTime = chrono::duration_cast<chrono::nanoseconds>(end - start).count()/1e9;
    cout << "Karatsuba: " << karatsubaTime << " s\n";
    //printPolynomial(p_C3, 2*SIZE-1);

    cout << SIZE << "," << simpleTime << "," << flattenedTime;
    cout << "," << karatsubaTime << "\n";

    verifyResults(p_C1, p_C2, p_C3);
    free (p_C1);
    free (p_C2);
    free (p_C3);

}



//CPU PART  //CPU PART  //CPU PART  //CPU PART  //CPU PART  //CPU PART  //CPU PART  //CPU PART  //CPU PART  //CPU PART  //CPU PART  //CPU PART  //CPU PART
//CPU PART  //CPU PART  //CPU PART  //CPU PART  //CPU PART  //CPU PART  //CPU PART  //CPU PART  //CPU PART  //CPU PART  //CPU PART  //CPU PART  //CPU PART
//CPU PART  //CPU PART  //CPU PART  //CPU PART  //CPU PART  //CPU PART  //CPU PART  //CPU PART  //CPU PART  //CPU PART  //CPU PART  //CPU PART  //CPU PART

//GPU PART  //GPU PART  //GPU PART  //GPU PART  //GPU PART  //GPU PART  //GPU PART  //GPU PART  //GPU PART  //GPU PART  //GPU PART  //GPU PART  //GPU PART
//GPU PART  //GPU PART  //GPU PART  //GPU PART  //GPU PART  //GPU PART  //GPU PART  //GPU PART  //GPU PART  //GPU PART  //GPU PART  //GPU PART  //GPU PART
//GPU PART  //GPU PART  //GPU PART  //GPU PART  //GPU PART  //GPU PART  //GPU PART  //GPU PART  //GPU PART  //GPU PART  //GPU PART  //GPU PART  //GPU PART


/**
 * @brief Calculates a coeficient for a given rank.
 * 
 * @param rank rank of calculated coeficient
 * @param p_A first polynomial
 * @param p_B second polynomial
 * @param lower first lower index (inclusive)
 * @param upper last upper index (exclusive)
 * @return COEFTYPE 
 */
__device__ COEFTYPE GPUmultiplicationCoeficient(size_t rank, 
                                                COEFTYPE* p_A, 
                                                COEFTYPE* p_B, 
                                                size_t lower, 
                                                size_t upper, 
                                                size_t size){
    COEFTYPE coef = 0;
    for (size_t j = 0; j < rank+1; j++){
        if (j >= size || rank-j >= size)
            continue;
        coef += p_A[j+lower]*p_B[rank-j+lower];
    }
    return coef;
}

/**
 * @brief Calculates a multiplication of polynomials by calculating each coeficient.
 * 
 * @param p_A first polynomial
 * @param p_B second polynomial
 * @param lower first lower index (inclusive)
 * @param upper last upper index (exclusive)
 * @return COEFTYPE* Returns new polynomial
 */
__global__ void GPUsimpleMultiplyPolynomials(COEFTYPE* p_A, 
                                            COEFTYPE* p_B, 
                                            COEFTYPE* p_C, 
                                            size_t lower, 
                                            size_t upper, 
                                            size_t size){
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=size)return;
    p_C[i] = GPUmultiplicationCoeficient(i,p_A,p_B, lower, upper, size);
    p_C[size + i] = GPUmultiplicationCoeficient(size + i,p_A,p_B, lower, upper, size);
    
}

COEFTYPE* GPUKaratsubaChoice(COEFTYPE* p_A, 
                            COEFTYPE* p_B, 
                            size_t lower, 
                            size_t upper){
    size_t s_upper = upper - lower;
    COEFTYPE *s_A = &p_A[lower];
    COEFTYPE *s_B = &p_B[lower];
    COEFTYPE *s_C = emptyInitialization(2*s_upper);
    COEFTYPE *d_A, *d_B, *d_C;
    cudaMalloc( (void**)&d_A, s_upper * sizeof(COEFTYPE) );
    cudaMalloc( (void**)&d_B, s_upper * sizeof(COEFTYPE) );
    cudaMalloc( (void**)&d_C, 2*s_upper * sizeof(COEFTYPE) );
    cudaMemcpy(d_A,s_A,sizeof(COEFTYPE)*s_upper,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,s_B,sizeof(COEFTYPE)*s_upper,cudaMemcpyHostToDevice);
    
    int n = (int)ceil(s_upper/(float)GPU_MAX_THREADS);
    GPUsimpleMultiplyPolynomials<<<n, GPU_MAX_THREADS>>>(d_A, d_B, d_C, 0, s_upper, s_upper);
        

    cudaMemcpy(s_C,d_C,sizeof(COEFTYPE)*s_upper*2,cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return s_C;
}

/**
 * @brief Calculates a multiplication of polynomials using a formula c[i+j]=a[i]*b[j]
 * 
 * @param p_A first polynomial
 * @param p_B second polynomial
 * @param lower first lower index (inclusive)
 * @param upper last upper index (exclusive)
 * @return COEFTYPE* Returns new polynomial
 */
__global__ void GPUflattenedMultiplyPolynomials(COEFTYPE* d_A, 
                                                COEFTYPE* d_B, 
                                                COEFTYPE* d_C,
                                                int size=GPU_MAX_THREADS,
                                                int blockOffset=0,
                                                int threadOffset=0){
    //__shared__ int calculations[1024];                                              
    int i = blockOffset+blockIdx.x;
    int j = threadOffset+threadIdx.x;

    //calculations[j] = d_C[i+j];
    //printf("%d : %d\n",i,j);
    if(size<=GPU_MAX_THREADS)
        atomicAdd(&(d_C[i+j]), d_A[i]*d_B[j]);
    else{
        int k=i, l=j;
        /*int doubleSize = 2*size;
        COEFTYPE a = d_A[k];
        COEFTYPE b = d_B[k];
        while(k+l<doubleSize){
            atomicAdd(&(d_C[k+l]), a*d_B[l]);
            atomicAdd(&(d_C[k+l]), d_A[l]*b);
            l+=GPU_MAX_THREADS;
            if (l >= size){
                l = j;
                k+=GPU_MAX_THREADS;
                a = d_A[k];
                b = d_B[k];
            }
        }*/
        for(    k=i ; k<size ; k+=GPU_MAX_THREADS){
            COEFTYPE a = d_A[k];
            //COEFTYPE b = d_B[k];
            for(l=j ; l<size ; l+=GPU_MAX_THREADS){
                atomicAdd(&(d_C[k+l]), a*d_B[l]);
                //atomicAdd(&(d_C[k+l]), d_A[l]*b);
            }
        }
    }
}

void GPUflattenedRunner(COEFTYPE* d_A, 
                            COEFTYPE* d_B, 
                            COEFTYPE* d_C){
    if(SIZE <= GPU_MAX_THREADS)
        GPUflattenedMultiplyPolynomials<<<SIZE,SIZE>>>(d_A, d_B, d_C);
    else{
        GPUflattenedMultiplyPolynomials<<<GPU_MAX_THREADS,GPU_MAX_THREADS>>>(d_A, d_B, d_C, SIZE);
        /*int k, l;
        for(    k=0 ; k<SIZE ; k+=GPU_MAX_THREADS){
            for(l=0 ; l<SIZE ; l+=GPU_MAX_THREADS){
                GPUflattenedMultiplyPolynomials<<<GPU_MAX_THREADS,GPU_MAX_THREADS>>>(
                    d_A, d_B, d_C, 
                    GPU_MAX_THREADS, k, l);
            }
        }*/
    }
}

COEFTYPE* GPURunner(COEFTYPE* p_A, 
                    COEFTYPE* p_B,
                    COEFTYPE* p_C,
                    bool simple=1){
    COEFTYPE *d_A, *d_B, *d_C;

    cudaEvent_t start, stop;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    cudaMalloc( (void**)&d_A, SIZE * sizeof(COEFTYPE) );
    cudaMalloc( (void**)&d_B, SIZE * sizeof(COEFTYPE) );
    cudaMalloc( (void**)&d_C, 2*SIZE * sizeof(COEFTYPE) );
    cudaMemcpy(d_A,p_A,sizeof(COEFTYPE)*SIZE,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,p_B,sizeof(COEFTYPE)*SIZE,cudaMemcpyHostToDevice);
    cudaMemcpy(d_C,p_C,sizeof(COEFTYPE)*SIZE*2,cudaMemcpyHostToDevice);

    cudaEventRecord(start, 0);

    if(simple){
        int n = (int)ceil(SIZE/(float)GPU_MAX_THREADS);
        GPUsimpleMultiplyPolynomials<<<n, GPU_MAX_THREADS>>>(d_A, d_B, d_C, 0, SIZE, SIZE);
    } else{
        GPUflattenedRunner(d_A, d_B, d_C);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    cudaMemcpy(p_C,d_C,sizeof(COEFTYPE)*SIZE*2,cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    if(simple)
        cout << "SimpleGPU: ";
    else
        cout << "FlattenedGPU: ";
    cout << elapsedTime/1e3 << " s\n";
    //cout << elapsedTime/1e3 << ",";
    return p_C;
}

COEFTYPE* GPUKaratsubaRunner(COEFTYPE* p_A, 
                        COEFTYPE* p_B,
                        COEFTYPE* p_C){
    cudaEvent_t start, stop;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    p_C = karatsubaMultiplyPolynomials(p_A,p_B, 0, SIZE, true);
        
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    cout << "KaratsubaGPU: " << elapsedTime/1e3 << " s\n";
    //cout << elapsedTime/1e3 << "\n";
    return p_C;
}

COEFTYPE* GPURunner(COEFTYPE* p_A, 
                        COEFTYPE* p_B,
                        int choice = 0){
    COEFTYPE* p_C = initPolynomial(2 * SIZE, true);
    
    switch (choice)
    {
    case 1:
        GPURunner(p_A, p_B, p_C, 0);
        break;
    case 2:
        GPUKaratsubaRunner(p_A, p_B, p_C);
        break;
    default:
        GPURunner(p_A, p_B, p_C);
        break;
    }
}

void runOnGPU(COEFTYPE* p_A, COEFTYPE* p_B){
    COEFTYPE *p_C1, *p_C2, *p_C3; 
    //cout << SIZE << ",";
    p_C1 = GPURunner(p_A,p_B,0);
    //printPolynomial(p_C1, 2*SIZE-1);
    p_C2 = GPURunner(p_A,p_B,1);
    //printPolynomial(p_C2, 2*SIZE-1);
    p_C3 = GPURunner(p_A,p_B,2);
    //printPolynomial(p_C3, 2*SIZE-1);
    
    verifyResults(p_C1, p_C2, p_C3);
    free (p_C1);
    free (p_C2);
    free (p_C3);

}



//GPU PART  //GPU PART  //GPU PART  //GPU PART  //GPU PART  //GPU PART  //GPU PART  //GPU PART  //GPU PART  //GPU PART  //GPU PART  //GPU PART  //GPU PART
//GPU PART  //GPU PART  //GPU PART  //GPU PART  //GPU PART  //GPU PART  //GPU PART  //GPU PART  //GPU PART  //GPU PART  //GPU PART  //GPU PART  //GPU PART
//GPU PART  //GPU PART  //GPU PART  //GPU PART  //GPU PART  //GPU PART  //GPU PART  //GPU PART  //GPU PART  //GPU PART  //GPU PART  //GPU PART  //GPU PART

int main(void){
    srand (42);
    //for (size_t i = 0; i < 6; i++){
        COEFTYPE* p_A = initPolynomial(SIZE);
        COEFTYPE* p_B = initPolynomial(SIZE);
        //printPolynomial(p_A, SIZE);
        //printPolynomial(p_B, SIZE);

        //runOnCPU(p_A, p_B);
        runOnGPU(p_A, p_B);

        free (p_A);
        free (p_B);
        //SIZE *= 2;
    //}

    cout << "Exit success!\n";
    return 0;
}


//cudaDeviceSynchronize()
//ssh gpu-01
//exit
//gpu04-test
//blockIdx.x*blockDim.x+threadIdx.x;