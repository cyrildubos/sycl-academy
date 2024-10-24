#include <sycl/sycl.hpp>

int device_selector(const sycl::device &device) {
  return device.has(sycl::aspect::usm_device_allocations) ? 1 : -1;
}

int main() {
  auto queue = sycl::queue{device_selector};

  return EXIT_SUCCESS;
}