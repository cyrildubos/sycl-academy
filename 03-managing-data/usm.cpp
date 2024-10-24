#include <sycl/sycl.hpp>

constexpr size_t n = 64;

int main() {
  auto a = 18;
  auto b = 24;
  auto c = 0;

  std::cout << "a = " << a << '\n';
  std::cout << "b = " << b << '\n';

  auto queue = sycl::queue{};

  auto a_device = sycl::malloc_device<int>(1, queue);
  auto b_device = sycl::malloc_device<int>(1, queue);
  auto c_device = sycl::malloc_device<int>(1, queue);

  queue.memcpy(a_device, &a, sizeof(int)).wait();
  queue.memcpy(b_device, &b, sizeof(int)).wait();

  queue
      .submit([&](auto &handler) {
        handler.single_task([=] { c_device[0] = a_device[0] + b_device[0]; });
      })
      .wait();

  queue.memcpy(&c, c_device, sizeof(int)).wait();

  sycl::free(a_device, queue);
  sycl::free(b_device, queue);
  sycl::free(c_device, queue);

  std::cout << "a + b = " << c << '\n';

  return EXIT_SUCCESS;
}