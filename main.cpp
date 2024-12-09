#include <cassert>
#include <cstring>
#include <iomanip>
#include <iostream>

#include "integral.h"
#include "matrix.h"
#include "storage.h"
#include "shape.h"

template <typename T>
using int1e = std::vector<T>;
template <typename T>
using int2e = std::vector<T>;

size_t index_int2e(size_t sorb, size_t i, size_t j, size_t k, size_t l) {
  assert(i < sorb && j < sorb && k < sorb && l < sorb);
  return i * sorb * sorb * sorb + j * sorb * sorb + k * sorb + l;
}

template <typename T>
Matrix<T> diag_mat(Matrix<T> &A, size_t sorb) {
  Matrix<T> mat(sorb, sorb, 0.0);
  auto eigenvalues = A.eigen();
  for (size_t i = 0; i < sorb; i++) {
    mat(i, i) = eigenvalues[i];
  }
  return mat;
}

template <typename T>
Matrix<T> G_matrix(const Matrix<T> &P_density, const int2e<T> &U, size_t sorb) {
  /**
  G_ij = P_kl [(ij|lk) - 0.5(ik|lj)]
  **/
  Matrix<T> G(sorb, sorb, (T)0.0);

  T tmpj, tmpk;
  for (size_t i = 0; i < sorb; i++) {
    for (size_t j = 0; j < sorb; j++) {
      for (size_t k = 0; k < sorb; k++) {
        for (size_t l = 0; l < sorb; l++) {
          tmpj = U[index_int2e(sorb, i, j, l, k)];
          tmpk = (T)0.5 * U[index_int2e(sorb, i, k, l, j)];
          G(i, j) = G(i, j) + P_density(k, l) * (tmpj - tmpk);
        }
      }
    }
  }
  return G;
}

template <typename T>
Matrix<T> update_density_mat(const Matrix<T> &C_new, size_t sorb, size_t nele) {
  /**
  P_ij = 2 \sum_a^{N/2} C_ik * C_jk
  **/
  Matrix<T> P_new(sorb, sorb, (T)0.0);
  for (size_t i = 0; i < sorb; i++) {
    for (size_t j = 0; j < sorb; j++) {
      for (size_t k = 0; k < nele / 2; k++) {
        P_new(i, j) = P_new(i, j) + 2 * C_new(i, k) * C_new(j, k);
      }
    }
  }
  return P_new;
}

template <typename T>
T diff_D(const Matrix<T> &P1, const Matrix<T> &P2, size_t sorb) {
  T delta = (P1 - P2).pow().sum();
  return std::sqrt(delta / (sorb * sorb));
}

template <typename T>
T calc_E0(const Matrix<T> &H_core, const Matrix<T> &P, const int2e<T> &U,
          const size_t sorb) {
  auto Fock = G_matrix(P, U, sorb) + H_core;
  T e = ((H_core + Fock).transpose() * P).sum() * 0.5;
  return e;
}

int main() {
  /**
  LiH sorb = 11, nele = 4
  H2 sorb = 4, nele =2
  H2O sorb =13, nele = 10;
  **/
  size_t sorb = 11;
  size_t nele = 4;
  // size_t sorb = 4;
  // size_t nele = 2;
  using dtype = double;
  const std::string file = "./integral/S_LiH.txt";
  auto ovlp_int = integral::load<dtype>(sorb, file);
  const std::string file1 = "./integral/Hcore_LiH.txt";
  auto core_int = integral::load<dtype>(sorb, file1);
  const std::string file2 = "./integral/LiH.txt";
  auto U_int = integral::load<dtype>(sorb, file2);
  if (!(ovlp_int.size() == sorb * sorb && core_int.size() == ovlp_int.size())) {
    std::cout << "sorb error " << sorb << std::endl;
  }
  if (U_int.size() != sorb * sorb * sorb * sorb) {
    std::cout << "sorb error " << sorb << std::endl;
  }
  auto H_core = Matrix<dtype>(sorb, sorb, core_int.data());
  auto ovlp_mat = Matrix<dtype>(sorb, sorb, ovlp_int.data());

  // transformer matrix X S^(-0.5) error
  auto A = ovlp_mat.copy();
  auto S_diag = diag_mat(A, sorb);
  for (size_t i = 0; i < sorb; i++) {
    S_diag(i, i) = std::pow(S_diag(i, i), -0.5);
  }
  auto vec = A.transpose();  // F-order to C-order
  auto X_transfer = vec.matmul(S_diag.matmul(vec.transpose()));
  auto X_adjoint = X_transfer.transpose();

  // step 4 construct density matrix D_mat
  auto D_init = Matrix<dtype>(sorb, sorb, core_int.data());
  auto D_old = std::move(D_init);
  std::cout << D_old.data() << std::endl;

  auto x = st::Storage<dtype>(20);
  auto y = st::Storage(x, 10);
  auto shape1 = at::Shape({3, 4, 5, 10});
  std::cout << x.data_ptr() << std::endl;
  std::cout << y.data_ptr() << std::endl;
  std::cout << shape1 << std::endl;

  size_t count = 0;
  size_t max_count = 100;
  dtype E_old = 10;
  dtype delta_E = 10;
  dtype delta_D = 10;
  dtype threshold = 1e-12;
  auto E_orb = Matrix<dtype>();
  auto coeff = Matrix<dtype>();

  while (count < max_count) {
    // step 5 G_ij
    auto G_ij = G_matrix(D_old, U_int, sorb);
    // std::cout << "D_old\n" << D_old << std::endl;

    // step 6: construct Fock-matrix F = H_core + G_ij
    auto Fock_mat = H_core + G_ij;

    // step 7: F' = X+ @ F @ X
    // std::cout << "Fock\n" << Fock_mat << std::endl;
    // std::cout << "X+\n" << X_adjoint << std::endl;
    auto Fock_mat_x = X_adjoint.matmul(Fock_mat.matmul(X_transfer));
    // std::cout << "Fock_p\n" << Fock_mat_x << std::endl;

    // step 8: diagonalize F' => C', E
    auto C_eige = Fock_mat_x.copy();
    auto C_value = diag_mat(C_eige, sorb);

    // step 9: C = X @ C' (C' is eigenvetor)
    auto C_new = X_transfer.matmul(C_eige.transpose());
    // std::cout << "C_new\n" << C_new << std::endl;

    // step 10: Update density matrix using new coefficient (D_new)
    auto D_new = update_density_mat(C_new, sorb, nele);
    // std::cout << "D_new\n" << D_new << std::endl;

    auto E_ele = calc_E0(H_core, D_new, U_int, sorb);

    delta_D = diff_D(D_new, D_old, sorb);
    delta_E = count != 0 ? std::fabs(E_ele - E_old) : 0;
    E_old = E_ele;

    coeff = std::move(C_new);
    E_orb = std::move(C_value);

    std::cout << std::setprecision(12) << "Delta-D: " << delta_D << "  "
              << "SCF-energy: " << E_ele << " " << "Delta-E: " << delta_E
              << "\n";
    // std::cout << count <<"-th" << "\n===================" << std::endl;
    if (delta_D < 1e-6 && delta_E < threshold) {
      std::cout << "SCF-converge" << std::endl;
      break;
    }

    D_old = std::move(D_new);
    count += 1;
  }

  // AO to MO
  return 0;
}