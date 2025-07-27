#include "wisp_spc_ops.h"

#include <kaolin/csrc/ops/spc/point_utils.h>
#include <kaolin/csrc/ops/spc/spc.h>
#include <torch/torch.h>

#include "spc_ops/spc_ops.h"

namespace wisp_spc_ops
{
    torch::Tensor dilate_points(torch::Tensor points, int level)
    {
        auto tensor_option =
            torch::TensorOptions().dtype(torch::kInt16).device(points.device());
        torch::Tensor _x = torch::tensor({{1, 0, 0}}, tensor_option);
        torch::Tensor _y = torch::tensor({{0, 1, 0}}, tensor_option);
        torch::Tensor _z = torch::tensor({{0, 0, 1}}, tensor_option);
        points = torch::cat(
            {points + _x, points - _x, points + _y,
             points - _y, points + _z, points - _z,
             points + _x + _y, points + _x - _y, points + _x + _z,
             points + _x - _z, points + _y + _x, points + _y - _x,
             points + _y + _z, points + _y - _z, points + _z + _x,
             points + _z - _x, points + _z + _y, points + _z - _y,
             points + _x + _y + _z, points + _x + _y - _z, points + _x - _y + _z,
             points + _x - _y - _z, points - _x + _y + _z, points - _x + _y - _z,
             points - _x - _y + _z, points - _x - _y - _z},
            0);
        points = torch::clamp(points, 0, pow(2, level) - 1);

        auto unique =
            std::get<0>(torch::unique_dim(points.contiguous(), 0, true, true, true));

        auto morton = std::get<0>(
            torch::sort(spc_ops::points_to_morton(unique.contiguous()).contiguous()));

        points = spc_ops::morton_to_points(morton.contiguous());

        return points;
    }

    std::pair<torch::Tensor, torch::Tensor>
    pointcloud_to_octree(const torch::Tensor &pointcloud, int level,
                         const torch::Tensor &attributes, int dilate)
    {
        torch::Tensor points =
            spc_ops::quantize_points(pointcloud.contiguous().cuda(), level);

        for (int i = 0; i < dilate; i++)
        {
            points = dilate_points(points, level);
        }

        torch::Tensor unique, unique_keys, unique_counts;
        std::tie(unique, unique_keys, unique_counts) =
            torch::unique_dim(points.contiguous(), 0, true, true, true);

        torch::Tensor morton, keys;
        std::tie(morton, keys) =
            torch::sort(spc_ops::points_to_morton(unique.contiguous()).contiguous());

        points = spc_ops::morton_to_points(morton.contiguous());
        torch::Tensor octree =
            spc_ops::unbatched_points_to_octree(points, level, true);

        if (attributes.defined())
        {
            torch::Tensor att = torch::zeros_like(unique).to(torch::kFloat32);
            att = att.index_add_(0, unique_keys, attributes) /
                  unique_counts.unsqueeze(-1).to(torch::kFloat32);
            att = att.index_select(0, keys);
            return std::make_pair(octree, att);
        }

        return std::make_pair(octree, torch::Tensor());
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    octree_to_spc(torch::Tensor octree)
    {
        torch::Tensor lengths =
            torch::tensor({octree.size(0)}, torch::dtype(torch::kInt32));
        auto octree_info = kaolin::scan_octrees_cuda(octree, lengths);
        auto pyramid = std::get<1>(octree_info);
        auto prefix = std::get<2>(octree_info);
        torch::Tensor points = kaolin::generate_points_cuda(octree, pyramid, prefix);
        pyramid = pyramid[0];
        return std::make_tuple(points, pyramid, prefix);
    }
    
    // 根据设置的采样数，对射线在相交节点上的穿入点和穿出点之间均匀采样，返回每条射线上 num_samples 个采样点的深度值(N, num_samples)。
    torch::Tensor sample_from_depth_intervals(torch::Tensor depth_intervals,
                                              int num_samples)
    {
        torch::Device device = depth_intervals.device();
        // (N, num_samples)
        torch::Tensor steps = torch::arange(num_samples, device)
                                  .unsqueeze(0)
                                  .to(torch::kFloat32)
                                  .repeat({depth_intervals.size(0), 1});
        
        // 每一步都添加随机噪声([0,1]的均匀分布)。
        steps += torch::rand_like(steps);

        // 将不同采样数，都将其值归一化到[0,1]
        steps *= (1.0 / num_samples);

        // (N, num_samples)
        torch::Tensor samples =
            // tensor.slice(dim, start, end[不包含], step=1)
            depth_intervals.slice(-1, 0, 1) +
            // 对射线在相交节点上的穿入点和穿出点之间均匀采样。
            (depth_intervals.slice(-1, 1, 2) - depth_intervals.slice(-1, 0, 1)) *
                steps;

        return samples;
    }

    // 根据 num_samples 值，扩展 pack_boundary 中的 1 值的位置。 
    torch::Tensor expand_pack_boundary(torch::Tensor pack_boundary,
                                       int num_samples)
    {
        /*
         """Expands the pack boundaries according to the number of samples.

         Args:
             pack_boundary (torch.BoolTensor): pack boundaries [N]
             num_samples (int): Number of samples

         Returns:
             (torch.BoolTensor): pack boundaries of shape [N*num_samples]
         """
        */
        torch::Tensor bigpack_boundary =
            torch::zeros({pack_boundary.size(0) * num_samples},
                         torch::TensorOptions()
                             .dtype(torch::kBool)
                             .device(pack_boundary.device()));
        // 将非零值的对应位置设置为 true 。
        bigpack_boundary.index_put_({pack_boundary.nonzero() * num_samples}, true);
        return bigpack_boundary.to(torch::kInt);
    }
} // namespace wisp_spc_ops