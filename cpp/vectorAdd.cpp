/**
 * @file polynomial.cu
 * @author Dominik Šmejkal
 * @brief polynomial multiplication on cpu
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
#define SIZE 2560
#define KARATSUBA_EXPANSION_BOUND SIZE-1
#define COEFTYPE long long int

void printPolynomial(COEFTYPE* p, unsigned int size);

COEFTYPE* emptyInitialization(unsigned int size){
    COEFTYPE * polynomial = (COEFTYPE *)malloc(size * sizeof(COEFTYPE));
    if (polynomial == NULL) {
     fprintf(stderr, "Failed to allocate host vector!\n");
     exit(EXIT_FAILURE);
    }
    return polynomial;
}

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

COEFTYPE* simpleMultiplyPolynomials(COEFTYPE* p_A, COEFTYPE* p_B, size_t lower, size_t upper){
    size_t size = upper - lower;
    COEFTYPE* p_C = emptyInitialization(2 * size);
    for (size_t i = 0; i < 2 * size; i++){
        p_C[i] = multiplicationCoeficient(i,p_A,p_B, lower, upper);

    }
    
    return p_C;
}

//c[i+j]=a[i]*b[j]
//cudaDeviceSynchronize()
//ssh gpu-01
//exit
//gpu04-test
COEFTYPE* flattenedMultiplyPolynomials(COEFTYPE* p_A, COEFTYPE* p_B, size_t lower, size_t upper){
    size_t size = upper - lower;
    COEFTYPE* p_C = initPolynomial(2 * size, true);
    //#pragma omp parallel {
		//#pragma omp single
        #pragma omp parallel for \
	   	default(shared) \
	   	schedule(dynamic)
        for (size_t i = 0; i < size; i++){
            //#pragma omp task firstprivate(i) default(shared)
            for (size_t j = 0; j < size; j++)
                p_C[i+j] = p_A[i+lower]*p_B[j+lower];
        }
        //#pragma omp taskwait
	//}

    
    return p_C;
}

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

COEFTYPE* concat(COEFTYPE* base, COEFTYPE* attached, size_t offset, size_t baseSize, size_t attachedSize, bool inplace = false){
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

//( (E1*D1) * (x^n) + ((E1 + E0) * (D0 + D1) - (D1*E1) - (D0*E0)) * (x ^ n/2) + E0D0)
COEFTYPE* karatsubaMultiplyPolynomials(COEFTYPE* p_A, COEFTYPE* p_B, size_t lower, size_t upper){
    size_t size = upper - lower;
    COEFTYPE *l,*u,*x, *result, *sum_A, *sum_B, *x_sub_A, *x_sub_AB, *l_x;
    if (upper - lower <= KARATSUBA_EXPANSION_BOUND)
        return flattenedMultiplyPolynomials(p_A,p_B, lower, upper);

    size_t half = (upper + lower)/2;
    size_t halfSize = upper - half;
    l = karatsubaMultiplyPolynomials(p_A,p_B, lower, half);
    u = karatsubaMultiplyPolynomials(p_A,p_B, half, upper);
    sum_A = sumHalves(p_A, lower, half, upper);
    sum_B = sumHalves(p_B, lower, half, upper);

    x = karatsubaMultiplyPolynomials(sum_A, sum_B, 0, halfSize);
    x_sub_A = polynomialSubtract(x, l, size, size);
    x_sub_AB = polynomialSubtract(x_sub_A, u, size, size);
/*
    cout << lower << " to " << upper << "\n";
    cout << "sum_A: ";
    printPolynomial(sum_A, halfSize);
    cout << "sum_B: ";
    printPolynomial(sum_B, halfSize);
    cout << "x: ";
    printPolynomial(x, size);
    cout << "l: ";
    printPolynomial(l, size);
    cout << "x: ";
    printPolynomial(x_sub_AB, size);
    cout << "u: ";
    printPolynomial(u, size);
    */
    l_x = concat(l, x_sub_AB, halfSize, size, size);
    result = concat(l_x, u, size, 1.5*size, size);

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

COEFTYPE* multiplyPolynomials(COEFTYPE* p_A, COEFTYPE* p_B, bool karatsuba = false, bool flattened=false){
    if (!karatsuba)
        return simpleMultiplyPolynomials(p_A,p_B, 0, SIZE);
    else
        return karatsubaMultiplyPolynomials(p_A,p_B, 0, SIZE);
}

void printPolynomial(COEFTYPE* p, unsigned int size){
    for (unsigned int i = 0; i < size; i++){
        cout << p[i] << ", ";
    }
    cout << "\n";

}

int main(void){
    srand (42);
    COEFTYPE* p_A = initPolynomial(SIZE);
    COEFTYPE* p_B = initPolynomial(SIZE);
    /*cout << "p_A: ";
    printPolynomial(p_A, SIZE);
    cout << "p_B: ";
    printPolynomial(p_B, SIZE);
*/
    COEFTYPE* p_C;
    auto start = chrono::steady_clock::now();
    p_C = multiplyPolynomials(p_A,p_B);
    auto end = chrono::steady_clock::now();
    cout << "Simple: " << chrono::duration_cast<chrono::nanoseconds>(end - start).count() << "\n";
    //printPolynomial(p_C, 2*SIZE-1);
    free (p_C);
        
    
    start = chrono::steady_clock::now();
    p_C = multiplyPolynomials(p_A,p_B,false,true);
    end = chrono::steady_clock::now();
    cout << "Flattened: " << chrono::duration_cast<chrono::nanoseconds>(end - start).count() << "\n";
    //printPolynomial(p_C, 2*SIZE-1);
    free (p_C);

    start = chrono::steady_clock::now();
    p_C = multiplyPolynomials(p_A,p_B,true);
    end = chrono::steady_clock::now();
    cout << "Karatsuba: " << chrono::duration_cast<chrono::nanoseconds>(end - start).count() << "\n";
    //printPolynomial(p_C, 2*SIZE-1);
    free (p_C);

    free (p_A);
    free (p_B);
    
    return 0;
}