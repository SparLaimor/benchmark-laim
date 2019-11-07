#include "../benchmarkLaim.hpp"
#include <cmath>
#include <omp.h>

#include <iostream>


const int MAX_MATRIX_SIZE = 15000;
const int NUMBER_OF_REPEAT = 8;
const int64_t TIME_LIMIT = 10000000; // 10 sec


class SortHandler {
    private:
        double *array_copy = nullptr;
    public:

    void setup(double *array, int n) {
        if (array_copy == nullptr) {
            array_copy = new double[n];
            std::copy(array, array + n, array_copy);
        } else {
            std::copy(array_copy, array_copy + n, array);
        }
    }

    ~SortHandler() {
        delete [] array_copy;
    }
};


class MatrixHandler {
    private:
        double **matrix_copy = nullptr;
        int size;
    public:

    void setup(double **matrix, int n) {
        if (matrix_copy == nullptr) {
            matrix_copy = new double*[n];
            size = n;
            for (int i = 0; i < n; ++i) { 
                matrix_copy[i] = new double[n];  
                std::copy(matrix[i], matrix[i] + n, matrix_copy[i]);
            }
        } else {
            // setup given data to matrix
            for (int i = 0; i < n; ++i) std::copy(matrix_copy[i], matrix_copy[i] + n, matrix[i]);
        }
    }

    ~MatrixHandler() {
        if (matrix_copy == nullptr) return;
        for (int i = 0; i < size; ++i) delete [] matrix_copy[i];
        delete [] matrix_copy; 
    }
};


double myFunc(double i, double j) {
    return (4 * i * i - 3 * j * j) * cos(i * i + 6 * j) + cbrt(i + 2 * j);
}

void fillMatrix(double **matrix, const int n) {
    for(int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            matrix[i][j] = myFunc(i, j);
}

void fillMatrixParallel(double **matrix, const int n) {
    #pragma omp parallel for
    for(int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            matrix[i][j] = myFunc(i, j);
}

void matrixVectorMultiply(double **matrix, const double *vector, double *resultVector, const int n) {
    for(int i = 0; i < n; ++i) {
        resultVector[i] = 0;
        for (int j = 0; j < n; ++j)  
            resultVector[i] += matrix[i][j] * vector[j];
    }
}

void matrixVectorMultiplyParallel(double **matrix, const double *vector, double *resultVector, const int n) {
    int i, j;
    #pragma omp parallel for private (j)
    for(i = 0; i < n; ++i) {
        resultVector[i] = 0;
        for (j = 0; j < n; ++j)  
            resultVector[i] += matrix[i][j] * vector[j];
    }
}

void matrixMultiply(double **matrixA, double **matrixB, double **resultMatrix, const int n) {
    for(int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            resultMatrix[i][j] = 0;
            for (int k = 0; k < n; ++k)  
                resultMatrix[i][j] +=  matrixA[i][k] * matrixB[i][j];
        }
    }
}

void matrixMultiplyParallel(double **matrixA, double **matrixB, double **resultMatrix, const int n) {
    int i, j, k;
    #pragma omp parallel for private (j, k)
    for(i = 0; i < n; ++i) {
        for (j = 0; j < n; ++j) {
            resultMatrix[i][j] = 0;
            for (k = 0; k < n; ++k)  
                resultMatrix[i][j] +=  matrixA[i][k] * matrixB[i][j];
        }
    }
}

void integral(const double sX, const double eX, const double sY, const double eY, const double h, double &res) {
    double lX = eX - sX;
    double lY = eY - sY;
    double x, y;

    res = 0;
    for(int i = 0; i < static_cast<int>(lX / h); ++i) {
        for(int j = 0; j < static_cast<int>(lY / h); ++j) {
            x = sX + h * (0.5 + i);
            y = sY + h * (0.5 + j);
            res += h * h * myFunc(x, y) / (lX * lY);
        }
    }
}

void integralParallelRow(const double sX, const double eX, const double sY, const double eY, const double h, double &res) {
    double lX = eX - sX;
    double lY = eY - sY;
    double x, y;
    int i, j;

    res = 0;
    #pragma omp parallel for private (x, y, j) reduction (+: res)
    for(i = 0; i < static_cast<int>(lX / h); ++i) {
        for(j = 0; j < static_cast<int>(lY / h); ++j) {
            x = sX + h * (0.5 + i);
            y = sY + h * (0.5 + j);
            res += h * h * myFunc(x, y) / (lX * lY);
        }
    }
}

void integralParallelColumn(const double sX, const double eX, const double sY, const double eY, const double h, double &res) {
    double lX = eX - sX;
    double lY = eY - sY;
    double x, y;
    int i, j;

    res = 0;
    for(i = 0; i < static_cast<int>(lX / h); ++i) {
        #pragma omp parallel for private (x, y, j) reduction (+: res)
        for(j = 0; j < static_cast<int>(lY / h); ++j) {
            x = sX + h * (0.5 + i);
            y = sY + h * (0.5 + j);
            res += h * h * myFunc(x, y) / (lX * lY);
        }
    }
}

template<typename T>
void oddEvenSort(T *array, int n)
{
    int upper_bound = n / 2;
    if (n % 2 == 0) upper_bound -= 1;
    for(int i = 0; i < n; ++i) {
        if(i % 2 == 0) {
            for (int j = 0; j < n / 2; ++j) {
                if (array[2 * j] > array[2 * j + 1]) 
                    std::swap(array[2 * j], array[2 * j + 1]);
            }
        } else {
            for (int j = 0; j < upper_bound; ++j) {
                if (array[2 * j + 1] > array[2 * j + 2]) 
                    std::swap(array[2 * j + 1], array[2 * j + 2]);
            }
        }
    }
}

template<typename T>
void oddEvenSortParallel(T *array, int n)
{
    int upper_bound = n / 2;
    if (n % 2 == 0) upper_bound -= 1;
    for(int i = 0; i < n; ++i) {
        if(i % 2 == 0) {
            #pragma omp parallel for
            for (int j = 0; j < n / 2; ++j) {
                if (array[2 * j] > array[2 * j + 1]) 
                    std::swap(array[2 * j], array[2 * j + 1]);
            }
        } else {
            #pragma omp parallel for
            for (int j = 0; j < upper_bound; ++j) {
                if (array[2 * j + 1] > array[2 * j + 2]) 
                    std::swap(array[2 * j + 1], array[2 * j + 2]);
            }
        }
    }
}

double _floydAlgorithmMin(double a, double b) {
    double result = (a < b) ? a : b;
    if(a < 0 && b >= 0) result = b;
    if(b < 0 && a >= 0) result = a;
    if(a < 0 && b < 0)  result = -1;
    return result;
}

void floydAlgorithm(double **matrix, int n) {
    for(int k = 0; k < n; ++k)
        for(int i = 0; i < n; ++i)
            for(int j = 0; j < n; ++j) {
                 matrix[i][j] = _floydAlgorithmMin(matrix[i][j], matrix[i][k] + matrix[k][j]);
            }
}

void floydAlgorithmParallel(double **matrix, int n) {
    double t1, t2;
    for(int k = 0; k < n; ++k) {
        #pragma omp parallel for private (t1, t2)
        for(int i = 0; i < n; ++i) {
            for(int j = 0; j < n; ++j) {
                if (matrix[i][k] != -1 && matrix[k][j] != -1) {
                    t1 = matrix[i][j];
                    t2 = matrix[i][k] + matrix[k][j];
                    matrix[i][j] = _floydAlgorithmMin(t1, t2);
                }
            }
        }
    }
}

void _init(double**& matrix) {
    matrix = new double*[MAX_MATRIX_SIZE];

    for (int i = 0; i < MAX_MATRIX_SIZE; ++i) { 
        matrix[i] = new double[MAX_MATRIX_SIZE]; 
    }   
}

void _init(double*& vector) {
    vector = new double[MAX_MATRIX_SIZE];
}

void _clear(double**& matrix) {
    for (int i = 0; i < MAX_MATRIX_SIZE; ++i) {
        delete [] matrix[i];
    }
    delete [] matrix; 
}

void _clear(double*& vector) {
    delete [] vector;
}

int main() { 
    Bench::BenchmarkReport report("../benchmark_result/dataset.js");

    Bench::Benchmark<int,       void (*)(double**, int)>                                    b1("Fill Matrix", "n", NUMBER_OF_REPEAT, TIME_LIMIT);
    Bench::Benchmark<int,       void (*)(double**, const double*, double*, int)>            b2("Matrix Vector Multiply", "n", NUMBER_OF_REPEAT, TIME_LIMIT);
    Bench::Benchmark<int,       void (*)(double**, double**, double**, int)>                b3("Matrix Multiply", "n", NUMBER_OF_REPEAT, TIME_LIMIT);
    Bench::Benchmark<double,    void (*)(double, double, double, double, double, double&)>  b4("Integrate", "h", NUMBER_OF_REPEAT, TIME_LIMIT);
    Bench::Benchmark<int,       void (*)(double*, int)>                                     b5("Sort", "n", NUMBER_OF_REPEAT, TIME_LIMIT);
    Bench::Benchmark<int,       void (*)(double**, int)>                                    b6("Floyd Algorithm", "n", NUMBER_OF_REPEAT, TIME_LIMIT);

    b1.addFunction(&fillMatrix,             "fillMatrix",           &fillMatrixParallel,            "fillMatrixParallel");
    b2.addFunction(&matrixVectorMultiply,   "matrixVectorMultiply", &matrixVectorMultiplyParallel,  "matrixVectorMultiplyParallel");
    b3.addFunction(&matrixMultiply,         "matrixMultiply",       &matrixMultiplyParallel,        "matrixMultiplyParallel");
    b4.addFunction(&integral,               "integral",             &integralParallelRow,           "integralParallelRow",          &integralParallelColumn, "integralParallelColumn");
    b5.addFunction(&oddEvenSort<double>,    "oddEvenSort",          &oddEvenSortParallel<double>,   "oddEvenSortParallel");
    b6.addFunction(&floydAlgorithm,         "floydAlgorithm",       &floydAlgorithmParallel,        "floydAlgorithmParallel");
    
    double** matrix;        _init(matrix);
    double** result_matrix; _init(result_matrix);
    double* vector;         _init(vector);
    double* result_vector;  _init(result_vector);

    for (int n = 500; n < MAX_MATRIX_SIZE; n = (int) n * 1.2) {
        b1.runWithParam<Bench::EmptyHandler>(n, matrix, n);
        b3.runWithParam<Bench::EmptyHandler>(n, matrix, matrix, result_matrix, n);
        std::copy(matrix[0], matrix[0] + n, vector);
        b2.runWithParam<Bench::EmptyHandler>(n, matrix, vector, result_vector, n);
    }

    _clear(result_matrix);
    _clear(vector);
    _clear(result_vector);

    // setup matrix for floyd
    for(int i = 0; i < MAX_MATRIX_SIZE; ++i)
        for(int j = 0; j < MAX_MATRIX_SIZE; ++j) 
            matrix[i][j] = std::abs(i - j) * matrix[i][j];

    for (int n = 10; n < MAX_MATRIX_SIZE; n = (int) n * 1.2) {
        b6.runWithParam<MatrixHandler>(n, matrix, n);
    }

    _clear(matrix);

    double result_integral = 0;
    for (double h = 0.4; h > 0.000000000001; h /= 1.5) {
        b4.runWithParam<Bench::EmptyHandler>(h, 0, 16, 0, 16, h, result_integral);
    }

    vector = new double[MAX_MATRIX_SIZE * MAX_MATRIX_SIZE];
    for (int n = 0; n < MAX_MATRIX_SIZE * MAX_MATRIX_SIZE; ++n) vector[n] = myFunc(n, n);
    for (int n = 500; n < MAX_MATRIX_SIZE * MAX_MATRIX_SIZE; n = (int) n * 1.2) {
        b5.runWithParam<SortHandler>(n, vector, n);
    }
    delete [] vector;

    report.write(
        b1.toJS(),
        b2.toJS(),
        b3.toJS(),
        b4.toJS(),
        b5.toJS(),
        b6.toJS()
    );



    return 0;
}
