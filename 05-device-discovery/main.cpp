#include <sycl/sycl.hpp>

int device_selector(const sycl::device &device) {
  if (device.has(sycl::aspect::cpu)) {
    auto vendor = device.get_info<sycl::info::device::vendor>();

    if (vendor.find("Intel") != std::string::npos) {
      return 1;
    }
  }

  return -1;
}

int main() {
  auto queue = sycl::queue{device_selector};

  auto device = queue.get_device();

  auto vendor = device.get_info<sycl::info::device::vendor>();
  auto name = device.get_info<sycl::info::device::name>();
  auto driver_version = device.get_info<sycl::info::device::driver_version>();

  std::cout << "vendor = " << vendor << '\n';
  std::cout << "name = " << name << '\n';
  std::cout << "driver_version = " << driver_version << '\n';

  return EXIT_SUCCESS;
}
