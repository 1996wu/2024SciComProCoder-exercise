#pragma once
#include <random>
#include <chrono>


template <typename T>
double get_duration_nano(T t) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(t).count();
}

inline std::chrono::high_resolution_clock::time_point get_time() {
  return std::chrono::high_resolution_clock::now();
}

template <typename T>
auto rand01() {
  static std::mt19937 rng(42);
  static std::uniform_real_distribution<T> u0(-1.0, 1.0);
  return u0(rng);
}
