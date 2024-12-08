#pragma once
#include <cstddef>
#include <iostream>
#include <random>
#define OFFSET(i, j, M, N) ((i) * (N) + (j))  // Row-major order
// #define USE_MKL 0

#ifdef USE_MKL
#include <mkl.h>
#endif

template <typename T>
void matmul_kernel(const T* A, const T* B, T* C, size_t M, size_t N, size_t L) {
  // A: [M, N] B:[N, L] C[M, L]
  for (size_t i = 0; i < M; i++) {
    for (size_t k = 0; k < N; k += 4) {
      for (size_t j = 0; j < L; j++) {
        // C[i, j] = A[i, k] * B[K, j]
        // C[OFFSET(i, j, M, L)] += A[OFFSET(i, k, N, L)] * B[OFFSET(k, j, N,
        // L)];
        T tmp0 = A[OFFSET(i, k + 0, N, L)] * B[OFFSET(k + 0, j, N, L)];
        T tmp1 = A[OFFSET(i, k + 1, N, L)] * B[OFFSET(k + 1, j, N, L)];
        T tmp2 = A[OFFSET(i, k + 2, N, L)] * B[OFFSET(k + 2, j, N, L)];
        T tmp3 = A[OFFSET(i, k + 3, N, L)] * B[OFFSET(k + 3, j, N, L)];
        C[OFFSET(i, j, M, L)] += tmp0 + tmp1 + tmp2 + tmp3;
      }
    }
  }
}

template <typename T>
void matmul_impl(const T* A, const T* B, T* C, size_t M, size_t N, size_t L) {
  // A[M, N], B[N, L] => C[M, L]
#ifdef USE_MKL
  if constexpr (std::is_same_v<T, float>) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, L, N, 1.0f, A, N,
                B, L, 0.0f, C, L);
  } else if (std::is_same_v<T, double>) {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, L, N, 1.0, A, N,
                B, L, 0.0, C, L);
  }
#else
  constexpr size_t blockSize = 32;
  if (false) {
#pragma omp parallel for
    for (size_t i = 0; i < M; ++i) {
      for (size_t j = 0; j < L; ++j) {
        T tmp = 0.0;
        for (size_t k = 0; k < N; ++k) {
          // C[i, j] = A[i, k] + B[k, j]
          tmp += A[OFFSET(i, k, M, N)] * B[OFFSET(k, j, M, L)];
        }
        C[OFFSET(i, j, M, L)] = tmp;
      }
    }
  } else {
    size_t group_M = (M - 1) / blockSize + 1;
    size_t group_N = (N - 1) / blockSize + 1;
    size_t group_L = (L - 1) / blockSize + 1;
    // A[M, N], B[N, L] => C[M, L]
#pragma omp parallel for collapse(2)
    for (size_t i = 0; i < group_M; i++) {
      for (size_t j = 0; j < group_L; j++) {
        T block_A[blockSize * blockSize];
        T block_B[blockSize * blockSize];
        T block_C[blockSize * blockSize] = {0};
        for (size_t k = 0; k < group_N; k++) {
          // Load A block into block_A
          for (size_t ii = 0; ii < blockSize; ii++) {
            for (size_t kk = 0; kk < blockSize; kk++) {
              size_t row = i * blockSize + ii;
              size_t col = k * blockSize + kk;
              if (row < M && col < N) {
                block_A[ii * blockSize + kk] = A[OFFSET(row, col, M, N)];
              } else {
                block_A[ii * blockSize + kk] = 0;
              }
            }
          }
          // Load B block into block_B
          for (size_t kk = 0; kk < blockSize; kk++) {
            for (size_t jj = 0; jj < blockSize; jj++) {
              size_t row = k * blockSize + kk;
              size_t col = j * blockSize + jj;
              if (row < N && col < L) {
                block_B[kk * blockSize + jj] = B[OFFSET(row, col, N, L)];
              } else {
                block_B[kk * blockSize + jj] = 0;
              }
            }
          }
          matmul_kernel(block_A, block_B, block_C, blockSize, blockSize,
                        blockSize);
        }
        // load C-block into C
        for (size_t _i = 0; _i < blockSize; _i++) {
          for (size_t _j = 0; _j < blockSize; _j++) {
            size_t row = i * blockSize + _i;
            size_t col = j * blockSize + _j;
            if (row < M && col < L) {
              C[OFFSET(row, col, M, L)] = block_C[_i * blockSize + _j];
            }
          }
        }
      }
    }
  }
#endif
}

#ifdef USE_MKL
template <typename T>
void dot_mkl(const T* A, const T* X, T* C, size_t M, size_t N) {
  if constexpr (std::is_same_v<T, float>) {
    for (size_t i = 0; i < M; i++) {
      C[i] = cblas_sdot(N, A + i * N, 1, X, 1);
    }
  }
  {
    for (size_t i = 0; i < M; i++) {
      C[i] = cblas_ddot(N, A + i * N, 1, X, 1);
    }
  }
}
#endif

template <typename T>
void dot_impl(const T* A, const T* B, T* C, size_t M, size_t N) {
#ifdef USE_MKL
  dot_mkl(A, B, C, M, N);
#else
#pragma omp parallel for
  for (size_t i = 0; i < M; i++) {
    T tmp = 0.0;
    for (size_t j = 0; j < N; j++) {
      auto offset = i * N + j;
      tmp += A[offset] * B[j];  //
    }
    C[i] = tmp;
  }
#endif
}

template <typename T>
void transpose_impl(const T* A, T* B, size_t M, size_t N) {
  // A:[M, N] B[N, M]
  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < N; j++) {
      B[j * M + i] = A[i * N + j];
    }
  }
}

#ifdef USE_MKL
template <typename T>
int syev_symm(int N, T* A, T* eigenvalues) {
  // symmetry matrix
  int lda = N;
  int info;
  int lwork;
  T wkopt;
  T* work;
  lwork = -1;
  if constexpr (std::is_same_v<T, float>) {
    ssyev("Vectors", "Upper", &N, A, &lda, eigenvalues, &wkopt, &lwork, &info);
  } else {
    dsyev("Vectors", "Upper", &N, A, &lda, eigenvalues, &wkopt, &lwork, &info);
  }
  lwork = (int)(wkopt);
  work = new T[lwork];
  if constexpr (std::is_same_v<T, float>) {
    ssyev("Vectors", "Upper", &N, A, &lda, eigenvalues, work, &lwork, &info);
  } else {
    dsyev("Vectors", "Upper", &N, A, &lda, eigenvalues, work, &lwork, &info);
  }
  if (info != 0) {
    std::cerr << "Error in dsyev: info = " << info << std::endl;
    exit(-1);
  }
  delete[] work;
  work = nullptr;
  return 0;
}
#endif