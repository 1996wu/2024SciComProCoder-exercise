#include "shape.h"

#include <cassert>

namespace at {
Shape::Shape(std::initializer_list<size_t> dims) : dims_(dims) {
  assert(dims.size() < MAX_DIM);
}

Shape::Shape(size_t* dims, size_t dim_)
    : dims_(std::vector(dims, dims + dim_)) {
  assert(dim_ < MAX_DIM);
}

Shape::Shape(const Shape& other) : dims_(other.dims_) {
  assert(other.ndim() < MAX_DIM);
}

Shape::Shape(Shape&& other) : dims_(std::move(other.dims_)) {}

bool Shape::operator==(const Shape& other) const {
  if (this->ndim() != other.ndim()) return false;
  size_t i = 0;
  for (; i < dims_.size() && dims_[i] == other.dims_[i]; ++i);
  return i == dims_.size();
}

std::ostream& operator<<(std::ostream& os, const Shape& s) {
  os << '(' << s[0];
  for (size_t i = 1; i < s.ndim(); ++i) os << ", " << s[i];
  os << ")";
  return os;
}

}  // namespace at
