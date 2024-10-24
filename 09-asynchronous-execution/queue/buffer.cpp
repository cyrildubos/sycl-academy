#include <sycl/sycl.hpp>

constexpr size_t n = 1'024;

int main() {
  float a[n];
  float b[n];
  float c[n];

  for (auto i = 0; i < n; ++i) {
    a[i] = static_cast<float>(i);
    b[i] = static_cast<float>(i);
    c[i] = 0.0f;
  }

  auto queue = sycl::queue{};

  auto a_buffer = sycl::buffer<float>{a, sycl::range{n}};
  auto b_buffer = sycl::buffer<float>{b, sycl::range{n}};
  auto c_buffer = sycl::buffer<float>{c, sycl::range{n}};

  queue.submit([&](auto &handler) {
    auto a_accessor = sycl::accessor{a_buffer, handler, sycl::read_only};
    auto b_accessor = sycl::accessor{b_buffer, handler, sycl::read_only};
    auto c_accessor = sycl::accessor{c_buffer, handler, sycl::write_only};

    handler.parallel_for(sycl::range{n}, [=](sycl::id<1> id) {
      c_accessor[id] = a_accessor[id] + b_accessor[id];
    });
  });

  queue.wait();

  for (auto i = 0; i < n; ++i) {
    assert(c[i] == static_cast<float>(i));
  }

  return EXIT_SUCCESS;
}