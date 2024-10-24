#include <sycl/sycl.hpp>

constexpr size_t n = 512;

int main() {
  float a[n * n];
  float b[n * n];

  for (auto i = 0; i < n * n; ++i) {
    a[i] = static_cast<float>(i);
    b[i] = 0.0f;
  }

  auto queue = sycl::queue{};

  auto global_range = sycl::range{n, n};
  auto local_range = sycl::range{8, 8};

  auto nd_range = sycl::nd_range{global_range, local_range};

  auto a_buffer = sycl::buffer{a, global_range};
  auto b_buffer = sycl::buffer{b, global_range};

  queue.submit([&](auto &handler) {
    auto a_accessor = sycl::accessor{a_buffer, handler, sycl::read_only};
    auto b_accessor = sycl::accessor{b_buffer, handler, sycl::write_only};

    auto accessor = sycl::local_accessor<float, 2>{local_range, handler};

    handler.parallel_for(nd_range, [=](sycl::nd_item<2> nd_item) {
      auto global_id = nd_item.get_global_id();
      auto local_id = nd_item.get_local_id();

      accessor[sycl::id{local_id[1], local_id[0]}] = a_accessor[global_id];

      sycl::group_barrier(nd_item.get_group());

      auto offset = global_id - local_id;

      auto id = sycl::id{offset[1], offset[0]} + local_id;

      b_accessor[id] = accessor[local_id];
    });
  });

  queue.wait();

  for (auto i = 0; i < n; ++i) {
    for (auto j = 0; j < n; ++j) {
      assert(b[i * n + j] == a[j * n + i]);
    }
  }

  return EXIT_SUCCESS;
}