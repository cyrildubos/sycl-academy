#include <sycl/sycl.hpp>

constexpr size_t n = 1'024;

int main() {
  float a[n];
  float b[n];
  float c[n];
  float d[n];

  for (auto i = 0; i < n; ++i) {
    a[i] = static_cast<float>(i);
    b[i] = static_cast<float>(i);
    c[i] = static_cast<float>(i);
    d[i] = 0.0f;
  }

  auto queue = sycl::queue{sycl::property::queue::in_order{}};

  auto a_device = sycl::malloc_device<float>(n, queue);
  auto b_device = sycl::malloc_device<float>(n, queue);
  auto c_device = sycl::malloc_device<float>(n, queue);
  auto d_device = sycl::malloc_device<float>(n, queue);

  queue.memcpy(a_device, a, n * sizeof(float));
  queue.memcpy(b_device, b, n * sizeof(float));
  queue.memcpy(c_device, c, n * sizeof(float));

  queue.parallel_for(sycl::range{n},
                     [=](sycl::id<1> id) { a_device[id] *= 2.0f; });

  queue.parallel_for(sycl::range{n},
                     [=](sycl::id<1> id) { b_device[id] += a_device[id]; });

  queue.parallel_for(sycl::range{n},
                     [=](sycl::id<1> id) { c_device[id] -= a_device[id]; });

  queue.parallel_for(sycl::range{n}, [=](sycl::id<1> id) {
    d_device[id] = b_device[id] + c_device[id];
  });

  queue.memcpy(d, d_device, n * sizeof(float)).wait();

  sycl::free(a_device, queue);
  sycl::free(b_device, queue);
  sycl::free(c_device, queue);
  sycl::free(d_device, queue);

  for (auto i = 0; i < n; ++i) {
    assert(d[i] == 2.0f * static_cast<float>(i));
  }

  return EXIT_SUCCESS;
}