#include <sycl/sycl.hpp>

constexpr float ratio = 0.5f;

constexpr size_t n = 1'024;
constexpr size_t m = n * ratio;

std::vector<sycl::device> get_two_devices() {
  auto devices = sycl::device::get_devices();

  if (devices.size() == 1) {
    return {devices[0], devices[0]};
  }

  return {devices[0], devices[1]};
}

int main() {
  float a[n];
  float b[n];
  float c[n];

  for (auto i = 0; i < n; ++i) {
    a[i] = static_cast<float>(i);
    b[i] = static_cast<float>(i);
    c[i] = 0.0f;
  }

  auto devices = get_two_devices();

  auto queue_0 = sycl::queue{devices[0]};
  auto queue_1 = sycl::queue{devices[1]};

  auto a_buffer_0 = sycl::buffer{a, sycl::range{m}};
  auto b_buffer_0 = sycl::buffer{b, sycl::range{m}};
  auto c_buffer_0 = sycl::buffer{c, sycl::range{m}};

  auto a_buffer_1 = sycl::buffer{a + m, sycl::range{n - m}};
  auto b_buffer_1 = sycl::buffer{b + m, sycl::range{n - m}};
  auto c_buffer_1 = sycl::buffer{c + m, sycl::range{n - m}};

  queue_0.submit([&](auto &handler) {
    auto a_accessor = sycl::accessor{a_buffer_0, handler, sycl::read_only};
    auto b_accessor = sycl::accessor{b_buffer_0, handler, sycl::read_only};
    auto c_accessor = sycl::accessor{c_buffer_0, handler, sycl::write_only};

    handler.parallel_for(sycl::range{m}, [=](sycl::id<1> id) {
      c_accessor[id] = a_accessor[id] + b_accessor[id];
    });
  });

  queue_1.submit([&](auto &handler) {
    auto a_accessor = sycl::accessor{a_buffer_1, handler, sycl::read_only};
    auto b_accessor = sycl::accessor{b_buffer_1, handler, sycl::read_only};
    auto c_accessor = sycl::accessor{c_buffer_1, handler, sycl::write_only};

    handler.parallel_for(sycl::range{n - m}, [=](sycl::id<1> id) {
      c_accessor[id] = a_accessor[id] + b_accessor[id];
    });
  });

  queue_0.wait();
  queue_1.wait();

  for (auto i = 0; i < n; ++i) {
    assert(c[i] == 2.0f * static_cast<float>(i));
  }

  return EXIT_SUCCESS;
}