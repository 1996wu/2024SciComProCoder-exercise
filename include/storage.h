#pragma once

#include <cstring>
#include <memory>

#include "allocator.h"

namespace st {

template <typename T>
class Storage {
 public:
  explicit Storage(size_t size);
  Storage(const Storage& other, size_t offset);
  Storage<T>(size_t size, T value);
  Storage<T>(const T* data, size_t size);

  explicit Storage<T>(const Storage& other) = default;
  explicit Storage<T>(Storage&& other) = default;
  ~Storage() = default;
  Storage& operator=(const Storage& othTer) = delete;

  T operator[](size_t idx) const { return dptr_[idx]; }
  T& operator[](size_t idx) { return dptr_[idx]; }
  size_t offset(void) const { return dptr_ - bptr_->data_; }

  T* data_ptr(void) const { return bptr_->data_; };

 private:
  struct Vdata {
    size_t version_;
    T data_[1];
  };

  // base pointer
  std::shared_ptr<Vdata> bptr_;
  // data pointer
  T* dptr_;
};
}  // namespace st

namespace st {

template <typename T>
Storage<T>::Storage(size_t size)
    : bptr_(Alloc::shared_allocate<Vdata>(size * sizeof(T) + sizeof(size_t))),
      dptr_(bptr_->data_) {
  bptr_->version_ = 0;
}

template <typename T>
Storage<T>::Storage(const Storage& other, size_t offset)
    : bptr_(other.bptr_), dptr_(other.dptr_ + offset) {}

template <typename T>
Storage<T>::Storage(size_t size, T value) : Storage(size) {
  std::fill(dptr_, dptr_ + size, value);
}

template <typename T>
Storage<T>::Storage(const T* data, size_t size) : Storage(size) {
  std::memcpy(dptr_, data, size * sizeof(T));
}

}  // namespace st