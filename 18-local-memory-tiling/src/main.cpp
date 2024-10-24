#include <sycl/sycl.hpp>

#include <benchmark.h>
#include <image_conv.h>

constexpr size_t filter_width = 11;

int main() {
  const auto input_file = "res/dogs.png";
  const auto output_file = "res/blurred_dogs.png";

  auto filter = util::generate_filter(util::filter_type::blur, filter_width);

  size_t halo = filter.half_width();

  auto input_image = util::read_image(input_file, halo);

  size_t width = input_image.width();
  size_t height = input_image.height();

  auto output_image = util::allocate_image(width, height, 4);

  try {
    auto queue = sycl::queue{};

    auto global_range = sycl::range{width, height};
    auto local_range = sycl::range{8, 8};

    auto nd_range = sycl::nd_range{global_range, local_range};

    auto stride = sycl::range{1, 4};

    auto filter_range = filter_width * stride;

    auto input_range = (global_range + 2 * halo) * stride;
    auto output_range = global_range * stride;

    auto scratchpad_range = local_range + 2 * halo;

    auto filter_buffer = sycl::buffer{filter.data(), filter_range};

    auto input_buffer = sycl::buffer{input_image.data(), input_range};
    auto output_buffer = sycl::buffer<float, 2>{output_range};

    output_buffer.set_final_data(output_image.data());

    auto filter_vector =
        filter_buffer.reinterpret<sycl::float4>(filter_range / stride);

    auto input_vector =
        input_buffer.reinterpret<sycl::float4>(input_range / stride);
    auto output_vector =
        output_buffer.reinterpret<sycl::float4>(output_range / stride);

    util::benchmark(
        [&] {
          queue.submit([&](auto &handler) {
            auto filter_accessor =
                sycl::accessor{filter_vector, handler, sycl::read_only};

            auto input_accessor =
                sycl::accessor{input_vector, handler, sycl::read_only};
            auto output_accessor =
                sycl::accessor{output_vector, handler, sycl::write_only};

            auto scratchpad_accessor = sycl::local_accessor<sycl::float4, 2>{
                scratchpad_range, handler};

            handler.parallel_for(nd_range, [=](sycl::nd_item<2> nd_item) {
              auto global_id = nd_item.get_global_id();
              auto local_id = nd_item.get_local_id();

              auto group_id = nd_item.get_group().get_group_id();

              auto offset = group_id * local_range;

              for (auto i = local_id[0]; i < scratchpad_range[0];
                   i += local_range[0]) {
                for (auto j = local_id[1]; j < scratchpad_range[1];
                     j += local_range[1]) {
                  scratchpad_accessor[i][j] =
                      input_accessor[offset + sycl::range{i, j}];
                }
              }

              sycl::group_barrier(nd_item.get_group());

              auto entry = sycl::float4{0.0f, 0.0f, 0.0f, 0.0f};

              for (size_t x = 0; x < filter_width; ++x) {
                for (size_t y = 0; y < filter_width; ++y) {
                  auto offset = sycl::id{x, y};

                  entry += filter_accessor[offset] *
                           scratchpad_accessor[local_id + offset];
                }
              }

              output_accessor[global_id] = entry;
            });
          });

          queue.wait();
        },
        100, "image_convolution");
  } catch (std::exception &exception) {
    std::cerr << "EXCEPTION: " << exception.what() << '\n';
  }

  util::write_image(output_image, output_file);

  return EXIT_SUCCESS;
}