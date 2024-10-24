#include <sycl/sycl.hpp>

int main() {
  auto queue = sycl::queue{};

  queue
      .submit([&](auto &handler) {
        auto stream = sycl::stream{128, 128, handler};

        handler.single_task([=] { stream << "Hello, World!\n"; });
      })
      .wait();

  return EXIT_SUCCESS;
}