#include <sycl/sycl.hpp>

int main() {
  try {
    auto exception_handler = [&](sycl::exception_list exception_list) {
      for (auto &exception : exception_list) {
        std::rethrow_exception(exception);
      }
    };

    auto queue = sycl::queue{exception_handler};

    auto buffer = sycl::buffer<int>{sycl::range{1}};

    queue.submit([&](auto &handler) {
      auto accessor =
          buffer.get_access(handler, sycl::range{2}, sycl::read_only);
    });

    queue.throw_asynchronous();

    return EXIT_SUCCESS;
  } catch (const sycl::exception &exception) {
    std::cout << "EXCEPTION: " << exception.what() << '\n';

    return EXIT_FAILURE;
  }
}