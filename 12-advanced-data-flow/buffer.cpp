#include <sycl/sycl.hpp>

// TODO

constexpr size_t n = 1'024;

int main() {
  float a[n];
  float c[n];

  for (auto i = 0; i < n; ++i) {
    a[i] = static_cast<float>(i);
    c[i] = 0.0f;
  }

  auto queue = sycl::queue{};

  auto a_buffer = sycl::buffer{a, sycl::range{n}};
  auto b_buffer = sycl::buffer<float>{sycl::range{n}};
  auto c_buffer = sycl::buffer<float>{sycl::range{n}};

  a_buffer.set_final_data(nullptr);
  c_buffer.set_final_data(c);

  queue.submit([&](auto &handler) {
    auto a_accessor = sycl::accessor{a_buffer, handler, sycl::read_only};
    auto b_accessor = sycl::accessor{b_buffer, handler, sycl::write_only};

    handler.parallel_for(sycl::range{n}, [=](sycl::id<1> id) {
      b_accessor[id] = 8.0f * a_accessor[id];
    });
  });

  queue.submit([&](auto &handler) {
    auto b_accessor = sycl::accessor{b_buffer, handler, sycl::read_only};
    auto c_accessor = sycl::accessor{c_buffer, handler, sycl::write_only};

    handler.parallel_for(sycl::range{n}, [=](sycl::id<1> id) {
      c_accessor[id] = b_accessor[id] / 2.0f;
    });
  });

  queue.wait();

  for (auto i = 0; i < n; ++i) {
    assert(c[i] == 4.0f * static_cast<float>(i));
  }

  return EXIT_SUCCESS;
}