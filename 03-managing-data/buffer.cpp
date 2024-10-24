#include <sycl/sycl.hpp>

constexpr size_t n = 64;

int main() {
  auto a = 18;
  auto b = 24;
  auto c = 0;

  auto queue = sycl::queue{};

  auto a_buffer = sycl::buffer<int>{&a, sycl::range{1}};
  auto b_buffer = sycl::buffer<int>{&b, sycl::range{1}};
  auto c_buffer = sycl::buffer<int>{&c, sycl::range{1}};

  queue
      .submit([&](auto &handler) {
        auto a_accessor = sycl::accessor{a_buffer, handler, sycl::read_only};
        auto b_accessor = sycl::accessor{b_buffer, handler, sycl::read_only};
        auto c_accessor = sycl::accessor{c_buffer, handler, sycl::write_only};

        handler.single_task(
            [=] { c_accessor[0] = a_accessor[0] + b_accessor[0]; });
      })
      .wait();

  assert(a + b == c);

  return EXIT_SUCCESS;
}