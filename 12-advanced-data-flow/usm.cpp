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

  auto a_device = sycl::malloc_device<float>(n, queue);
  auto b_device = sycl::malloc_device<float>(n, queue);
  auto c_device = sycl::malloc_device<float>(n, queue);

  queue.memcpy(a_device, a, n).wait();

  queue
      .parallel_for(sycl::range{n},
                    [=](sycl::id<1> id) { b_device[id] = 8.0f * a_device[id]; })
      .wait();

  queue
      .parallel_for(sycl::range{n},
                    [=](sycl::id<1> id) { c_device[id] = b_device[id] / 2.0f; })
      .wait();

  queue.memcpy(c, c_device, n).wait();

  sycl::free(a_device, queue);
  sycl::free(b_device, queue);
  sycl::free(c_device, queue);

  for (auto i = 0; i < n; ++i) {
    assert(c[i] == 4.0f * static_cast<float>(i));
  }

  return EXIT_SUCCESS;
}