/*******************************************************************************
 *  Copyright (C) 2009-2015 Intel Corporation. All Rights Reserved.
 *  The information and material ("Material") provided below is owned by Intel
 *  Corporation or its suppliers or licensors, and title to such Material
 *remains with Intel Corporation or its suppliers or licensors. The Material
 *contains proprietary information of Intel or its suppliers and licensors. The
 *Material is protected by worldwide copyright laws and treaty provisions. No
 *part of the Material may be copied, reproduced, published, uploaded, posted,
 *  transmitted, or distributed in any way without Intel's prior express written
 *  permission. No license under any patent, copyright or other intellectual
 *  property rights in the Material is granted to or conferred upon you, either
 *  expressly, by implication, inducement, estoppel or otherwise. Any license
 *  under such intellectual property rights must be express and approved by
 *Intel in writing.
 *
 ********************************************************************************
 */
/*
   SSYEV Example.
   ==============

   Program computes all eigenvalues and eigenvectors of a real symmetric
   matrix A:

     1.96  -6.49  -0.47  -7.20  -0.65
    -6.49   3.80  -6.39   1.50  -6.34
    -0.47  -6.39   4.17  -1.51   2.67
    -7.20   1.50  -1.51   5.70   1.80
    -0.65  -6.34   2.67   1.80  -7.10

   Description.
   ============

   The routine computes all eigenvalues and, optionally, eigenvectors of an
   n-by-n real symmetric matrix A. The eigenvector v(j) of A satisfies

   A*v(j) = lambda(j)*v(j)

   where lambda(j) is its eigenvalue. The computed eigenvectors are
   orthonormal.

   Example Program Results.
   ========================

 SSYEV Example Program Results

 Eigenvalues
 -11.07  -6.23   0.86   8.87  16.09

 Eigenvectors (stored columnwise)
  -0.30  -0.61   0.40  -0.37   0.49
  -0.51  -0.29  -0.41  -0.36  -0.61
  -0.08  -0.38  -0.66   0.50   0.40
   0.00  -0.45   0.46   0.62  -0.46
  -0.80   0.45   0.17   0.31   0.16
*/
#include <mkl.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
/* SSYEV prototype */
// extern void ssyev( char* jobz, char* uplo, int* n, float* a, int* lda,
//                 float* w, float* work, int* lwork, int* info );
// /* Auxiliary routines prototypes */
// extern void print_matrix( char* desc, int m, int n, float* a, int lda );

/* Parameters */
#define N 5
#define LDA N

void print_matrix(char* desc, int m, int n, float* a, int lda) {
  int i, j;
  printf("\n %s\n", desc);
  for (i = 0; i < m; i++) {
    for (j = 0; j < n; j++) printf(" %6.2f", a[i + j * lda]);
    printf("\n");
  }
}

/* Main program */
int main() {
  /* Locals */
  int n = N, lda = LDA, info, lwork;
  float wkopt;
  float* work;
  /* Local arrays */
  float w[N];
  float a[LDA * N] = {1.96f,  0.00f,  0.00f, 0.00f,  0.00f,  -6.49f, 3.80f,
                      0.00f,  0.00f,  0.00f, -0.47f, -6.39f, 4.17f,  0.00f,
                      0.00f,  -7.20f, 1.50f, -1.51f, 5.70f,  0.00f,  -0.65f,
                      -6.34f, 2.67f,  1.80f, -7.10f};
  /* Executable statements */
  printf(" SSYEV Example Program Results\n");
  /* Query and allocate the optimal workspace */
  lwork = -1;
  ssyev("Vectors", "Upper", &n, a, &lda, w, &wkopt, &lwork, &info);
  lwork = (int)wkopt;
  std::cout << lwork << std::endl;
  work = (float*)malloc(lwork * sizeof(float));
  /* Solve eigenproblem */
  ssyev("Vectors", "Upper", &n, a, &lda, w, work, &lwork, &info);
  /* Check for convergence */
  if (info > 0) {
    printf("The algorithm failed to compute eigenvalues.\n");
    exit(1);
  }
  /* Print eigenvalues */
  print_matrix("Eigenvalues", 1, n, w, 1);
  /* Print eigenvectors */
  print_matrix("Eigenvectors (stored columnwise)", n, n, a, lda);
  /* Free workspace */
  free((void*)work);
  exit(0);
} /* End of SSYEV Example */

/* Auxiliary routine: printing a matrix */
