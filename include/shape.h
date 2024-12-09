#pragma once

#include <initializer_list>
#include <ostream>
#include <vector>

using IndexArray = std::vector<size_t>;

constexpr int MAX_DIM = 32;

namespace at {

class Shape {
  // constructor
  public:
  Shape(std::initializer_list<size_t> dims);
  Shape(size_t* dims, size_t dim);
  Shape(const Shape& other);
  Shape(Shape&& other);
  ~Shape() = default;

  bool operator==(const Shape& other) const;

  size_t ndim(void) const { return dims_.size(); }
  size_t operator[](size_t idx) const { return dims_[idx]; }
  size_t& operator[](size_t idx) { return dims_[idx]; }

  friend std::ostream& operator<<(std::ostream& out, const Shape& s);

 private:
  IndexArray dims_;
};
}  // namespace at