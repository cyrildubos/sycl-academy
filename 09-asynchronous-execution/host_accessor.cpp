#include <sycl/sycl.hpp>

constexpr size_t n = 1'024;

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

        handler.parallel_for(sycl::range{n}, [=](sycl::id<1> id) {
          c_accessor[id] = a_accessor[id] + b_accessor[id];
        });
      })
      .wait();

  auto c_accessor = c_buffer.get_host_access(sycl::read_only);

  for (auto i = 0; i < n; ++i) {
    assert(c_accessor[i] == 2 * i);
  }

  return EXIT_SUCCESS;
}