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

  auto a_device = sycl::malloc_device<float>(n, queue);
  auto b_device = sycl::malloc_device<float>(n, queue);
  auto c_device = sycl::malloc_device<float>(n, queue);

  queue.memcpy(a_device, a, n * sizeof(float));
  queue.memcpy(b_device, b, n * sizeof(float));

  queue.wait();

  queue.parallel_for(sycl::range{n}, [=](sycl::id<1> id) {
    c_device[id] = a_device[id] + b_device[id];
  });

  queue.wait();

  queue.memcpy(c, c_device, n * sizeof(float));

  queue.wait();

  sycl::free(a_device, queue);
  sycl::free(b_device, queue);
  sycl::free(c_device, queue);

  for (auto i = 0; i < n; ++i) {
    assert(c[i] == static_cast<float>(i));
  }

  return EXIT_SUCCESS;
}