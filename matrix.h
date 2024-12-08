#pragma once
#include <omp.h>

#include <cassert>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>

#include "kernel.h"
#include "tools.h"

template <typename T>
class Vec {
 protected:
  size_t _size;
  T *_elem;

 public:
  static_assert(std::is_same<T, float>::value || std::is_same<T, double>::value,
                "Only support float, double");
  Vec(size_t n, T v) {
    _elem = new T[n];
    _size = n;
    for (size_t i = 0; i < n; i++) {
      _elem[i] = v;
    }
  }
  Vec(size_t n) {
    _elem = new T[n];
    _size = n;
    for (size_t i = 0; i < n; i++) {
      _elem[i] = rand01<T>();
    }
  }

  Vec(const Vec<T> &v1) : _size(v1._size), _elem(new T[v1._size]) {
    for (size_t i = 0; i < v1._size; i++) {
      _elem[i] = v1._elem[i];
    }
  }

  Vec(std::initializer_list<T> init) : _size(init.size()), _elem(new T[_size]) {
    size_t i = 0;
    for (const T &val : init) {
      _elem[i++] = val;
    }
  }

  Vec(T *value, size_t n) : _size(n), _elem(new T[n]) {
    assert(n <= _size && n > 0);
    for (size_t i = 0; i < n; i++) {
      _elem[i] = value[i];
    }
  }

  ~Vec() { delete[] _elem; }

  T &operator[](const size_t i) {
    assert(i < _size);
    return this->_elem[i];
  }

  const T &operator[](size_t i) const {
    assert(i < _size);
    return _elem[i];
  }

  T sum() {
    T s = 0.0;
    for (size_t i = 0; i < _size; i++) {
      s += _elem[i];
    }
    return s;
  }

  T min() {
    T m = _elem[0];
    for (size_t i = 0; i < _size; i++) {
      m = _elem[i] > m ? m : _elem[i];
    }
    return m;
  }

  T max() {
    T m = _elem[0];
    for (size_t i = 0; i < _size; i++) {
      m = _elem[i] > m ? _elem[i] : m;
    }
    return m;
  }

  const size_t size() const { return this->_size; }

  T *data() { return &_elem[0]; }
  const T *data() const { return &_elem[0]; }

  friend std::ostream &operator<<(std::ostream &os, const Vec<T> &vec) {
    for (size_t i = 0; i < vec._size; ++i) {
      os << vec._elem[i] << " ";
    }
    return os;
  }
};

template <typename T>
class Matrix {
 private:
  size_t _row;
  size_t _col;
  size_t _size;
  struct Vdata {
    size_t version_;
    T data_[1];
  };
  T *_data = nullptr;            // data pointer
  std::shared_ptr<Vdata> bptr_;  // base pointer
  char _order = 'C';

 public:
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "Only support float, double");
  Matrix() = default;

  Matrix(size_t m, size_t n) {
    _data = new T[n * m];
    _row = m;
    _col = n;
    _size = m * n;
    for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < n; j++) {
        _data[i * n + j] = rand01<T>();
      }
    }
  }
  Matrix(size_t m, size_t n, T v) {
    _data = new T[n * m];
    _row = m;
    _col = n;
    _size = m * n;
    for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < n; j++) {
        _data[i * n + j] = v;
      }
    }
  }

  Matrix(size_t m, size_t n, T *v) {
    _data = new T[m * n];
    _row = m;
    _col = n;
    _size = m * n;
    std::memcpy(_data, v, n * m * sizeof(T));
  }

  Matrix(size_t m, size_t n, T *begin, T *end) {
    _data = begin;
    _row = m;
    _col = n;
    _size = m * n;
  }

  Matrix(const Matrix &mat) noexcept {
    auto [_row, _col] = mat.size();
    _size = _row * _col;
    _data = new T[_size];
    std::memcpy(_data, mat.data(), _size * sizeof(T));
  }

  Matrix(Matrix &&other) noexcept {
    _row = other._row;
    _col = other._col;
    _size = other._size;
    _data = other._data;
    other._data = nullptr;
  }

  T &operator[](const size_t i) { return {this->_data[i]}; }
  const T &operator[](const size_t i) const { return {this->_data[i]}; }

  T &operator()(size_t i, size_t j) { return _data[i * _col + j]; }
  const T &operator()(size_t i, size_t j) const { return _data[i * _col + j]; }

  const T *data() const { return &_data[0]; };
  T *data() { return &_data[0]; };
  const auto row() const { return this->_row; }
  const auto col() const { return this->_col; }
  const auto size() const { return std::make_tuple(_row, _col); }

  const T &diag(size_t i) const {
    assert(i < _col && i < _row);
    return _data[i * _col + i];
  }

  T &diag(size_t i) {
    assert(i < _col && i < _row);
    return _data[i * _col + i];
  }

  Matrix operator*(const Matrix &mat) const {
    auto [M, N] = mat.size();
    assert(M == _row && N == _col);
    Matrix<T> result(M, N, 0.0);
    for (size_t i = 0; i < M; i++) {
      for (size_t j = 0; j < N; j++) {
        result(i, j) = (*this)(i, j) * mat(i, j);
      }
    }
    return result;
  }

  Matrix &operator*=(const Matrix &mat) {
    auto [M, N] = mat.size();
    assert(M == _row && N == _col);
    for (size_t i = 0; i < M; i++) {
      for (size_t j = 0; j < N; j++) {
        (*this)(i, j) *= mat(i, j);
      }
    }
    return (*this);
    ;
  }

  Matrix operator*(const T scaler) {
    size_t M = _row;
    size_t N = _col;
    Matrix<T> result(M, N, 0.0);
    for (size_t i = 0; i < M; i++) {
      for (size_t j = 0; j < N; j++) {
        result(i, j) = (*this)(i, j) * scaler;
      }
    }
    return result;
  }

  Matrix &operator*=(const T scaler) {
    size_t M = _row;
    size_t N = _col;
    for (size_t i = 0; i < M; i++) {
      for (size_t j = 0; j < N; j++) {
        (*this)(i, j) *= scaler;
      }
    }
    return *this;
  }

  Matrix &operator=(const Matrix &&other) noexcept {
    if (this == &other) {
      return *this;
    }
    _row = other._row;
    _col = other._col;
    _size = other._size;
    _data = other._data;
    other._data = nullptr;

    return *this;
  }

  Matrix operator+(const Matrix &mat) const {
    auto [M, N] = mat.size();
    assert(M == _row && N == _col);
    Matrix<T> result(M, N, 0.0);
    for (size_t i = 0; i < M; i++) {
      for (size_t j = 0; j < N; j++) {
        result(i, j) = (*this)(i, j) + mat(i, j);
      }
    }
    return result;
  }

  Matrix &operator+=(const Matrix<T> &mat) {
    auto [M, N] = mat.size();
    assert(M == _row && N == _col);
    for (size_t i = 0; i < M; i++) {
      for (size_t j = 0; j < N; j++) {
        (*this)(i, j) += mat(i, j);
      }
    }
    return (*this);
  }

  Matrix operator-(const Matrix<T> &mat) const{
    auto [M, N] = mat.size();
    assert(M == _row && N == _col);
    Matrix<T> result(M, N, 0.0);
    for (size_t i = 0; i < M; i++) {
      for (size_t j = 0; j < N; j++) {
        result(i, j) = (*this)(i, j) - mat(i, j);
      }
    }
    return result;
  }

  Matrix &operator-=(const Matrix<T> &mat) {
    auto [M, N] = mat.size();
    assert(M == _row && N == _col);
    for (size_t i = 0; i < M; i++) {
      for (size_t j = 0; j < N; j++) {
        (*this)(i, j) -= mat(i, j);
      }
    }
    return (*this);
  }

  Matrix &operator=(Matrix &&other) noexcept {
    if (this != &other) {
      delete[] _data;
      _row = other._row;
      _col = other._col;
      _size = other._size;
      _data = other._data;
      other._data = nullptr;
    }

    return *this;
  }

  Matrix matmul(const Matrix<T> &mat) {
    auto [N, L] = mat.size();  // [M, N, N, L]
    assert(N == _col);
    auto M = _row;
    auto result = Matrix<T>(M, L, 0.0);
    matmul_impl((*this).data(), mat.data(), result.data(), M, N, L);
    return result;
  }

  Matrix transpose() {
    auto [M, N] = (*this).size();
    auto result = Matrix<T>(N, M, 0.0);
    transpose_impl((*this).data(), result.data(), M, N);
    return result;
  }

  Matrix pow() {
    auto [M, N] = (*this).size();
    auto result = Matrix<T>(N, M, 0.0);
    for (size_t i = 0; i < M; i++) {
      for (size_t j = 0; j < N; j++) {
        result(i, j) = (*this)(i, j) * (*this)(i, j);
      }
    }
    return result;
  }

  bool check_conj(T threshold = 1e-8) {
    // bool flag = true;
    auto [M, N] = (*this).size();
    for (size_t i = 0; i < M; i++) {
      for (size_t j = 0; j < i; j++) {
        if (fabs((*this)(i, j) - (*this)(j, i)) > threshold) {
          return false;
        }
      }
    }
    return true;
  }

  void convert_order() {
    auto [M, N] = (*this).size();
    for (size_t i = 0; i < M; ++i) {
      for (size_t j = i + 1; j < N; ++j) {
        std::swap(_data[i * N + j], _data[j * M + i]);
      }
    }
    _order = 'F';
  }

  std::vector<T> eigen() {
    // TODO: dgeev interface
    /**
    notice: eigenvetors is F-order, change data
    **/
    auto [M, N] = (*this).size();
    // assert(M == N);
    if (!(*this).check_conj()) {
      std::cout << "only support conjugate matrix" << std::endl;
      exit(-1);
    }
    std::vector<T> eigenvalues(N, 0);
    // std::vector<T> data_m(N * N, 0);
    // std::memcpy(data_m.data(), (*this).data(), _size * sizeof(T));
    /* Executable statements */
    // syev_symm<T>(M, data_m.data(), eigenvalues.data());
    syev_symm<T>(M, (*this).data(), eigenvalues.data());
    return eigenvalues;
  }

  Vec<T> dot(const Vec<T> &vec) {
    assert(vec.size() == _col);
    Vec<T> result(_row, 0);
    dot_impl((*this).data(), vec.data(), result.data(), _row, _col);
    return result;
  }

  void mask(T threshold = 0.1) {
    for (size_t i = 0; i < _row; i++) {
      for (size_t j = 0; j < _col; j++) {
        (*this)(i, j) = fabs((*this)(i, j)) < threshold ? 0 : (*this)(i, j);
      }
    }
  }

  Matrix<T> copy() { return Matrix<T>(_row, _col, (*this).data()); }

  const T sum() const {
    T tmp = 0;
    // #pragma omp parallel for reduction(+ : tmp)
    for (size_t i = 0; i < _row; i++) {
      for (size_t j = 0; j < _col; j++) {
        tmp += (*this)(i, j);
      }
    }
    return tmp;
  }

  Matrix<T> &resize(size_t i, size_t j) {
    assert(i * j == _size);
    _row = i;
    _col = j;
    return *this;
  }

  const size_t numel() const { return _row * _col; }
  const size_t non_numel(T threshold = 1e-15) const {
    assert(threshold >= 0.0);
    size_t num = 0;
    for (size_t i = 0; i < _row; i++) {
      for (size_t j = 0; j < _col; j++) {
        if (fabs((*this)(i, j)) > threshold) {
          num += 1;
        };
      }
    }
    return num;
  }

  friend std::ostream &operator<<(std::ostream &os, const Matrix<T> &mat) {
    std::cout << "shape: (" << mat._row << " " << mat._col << ")\n";
    std::cout << std::setprecision(6);
    for (size_t i = 0; i < mat._row; i++) {
      for (size_t j = 0; j < mat._col; j++) {
        os << mat._data[i * mat._col + j] << " ";
      }
      os << "\n";
    }
    return os;
  };

  ~Matrix() { delete[] _data; }
};
