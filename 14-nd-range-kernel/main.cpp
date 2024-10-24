#include <sycl/sycl.hpp>

constexpr size_t n = 1'024;
constexpr size_t m = 128;

int main() {
  int a[n];
  int b[n];
  int c[n];

  for (auto i = 0; i < n; ++i) {
    a[i] = i;
    b[i] = i;
    c[i] = 0;
  }

  auto queue = sycl::queue{};

  auto a_buffer = sycl::buffer{a, sycl::range{n}};
  auto b_buffer = sycl::buffer{b, sycl::range{n}};
  auto c_buffer = sycl::buffer{c, sycl::range{n}};

  queue
      .submit([&](auto &handler) {
        auto a_accessor = sycl::accessor{a_buffer, handler, sycl::read_only};
        auto b_accessor = sycl::accessor{b_buffer, handler, sycl::read_only};
        auto c_accessor = sycl::accessor{c_buffer, handler, sycl::write_only};

        handler.parallel_for(sycl::nd_range{sycl::range{n}, sycl::range{m}},
                             [=](sycl::nd_item<1> item) {
                               auto global_id = item.get_global_id();

                               c_accessor[global_id] = a_accessor[global_id] +
                                                       b_accessor[global_id];
                             });
      })
      .wait();

  for (auto i = 0; i < n; ++i) {
    assert(c[i] == 2 * i);
  }

  return EXIT_SUCCESS;
}