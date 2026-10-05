#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace geometry {

struct Mesh {
	std::vector<std::array<float, 3>> positions;
	std::vector<uint32_t> triangles; // 3 indices per face
	std::vector<uint32_t> edges;     // 2 indices per edge 
};

inline Mesh makeOctahedron() {
	using V3 = std::array<double, 3>;

	const std::vector<V3> v = {
		{ 1.0,  0.0,  0.0}, {-1.0,  0.0,  0.0},
		{ 0.0,  1.0,  0.0}, { 0.0, -1.0,  0.0},
		{ 0.0,  0.0,  1.0}, { 0.0,  0.0, -1.0},
	};

	const auto dist = [&](size_t i, size_t j) {
		const double dx = v[i][0] - v[j][0];
		const double dy = v[i][1] - v[j][1];
		const double dz = v[i][2] - v[j][2];
		return std::sqrt(dx * dx + dy * dy + dz * dz);
	};

	// Длина ребра октаэдра с единичным радиусом описанной окружности
	const double edge_length = std::sqrt(2.0);
	const size_t n = v.size();

	const auto connected = [&](size_t i, size_t j) {
		return std::abs(dist(i, j) - edge_length) < 1e-6;
	};

	Mesh mesh;

	for (const V3& p : v) {
		mesh.positions.push_back({float(p[0]), float(p[1]), float(p[2])});
	}

	for (size_t i = 0; i < n; ++i) {
		for (size_t j = i + 1; j < n; ++j) {
			if (connected(i, j)) {
				mesh.edges.push_back(uint32_t(i));
				mesh.edges.push_back(uint32_t(j));
			}
		}
	}

	for (size_t i = 0; i < n; ++i) {
		for (size_t j = i + 1; j < n; ++j) {
			if (!connected(i, j)) continue;
			for (size_t k = j + 1; k < n; ++k) {
				if (!connected(i, k) || !connected(j, k)) continue;

				const V3 e1 = {v[j][0] - v[i][0], v[j][1] - v[i][1], v[j][2] - v[i][2]};
				const V3 e2 = {v[k][0] - v[i][0], v[k][1] - v[i][1], v[k][2] - v[i][2]};
				const V3 normal = {
					e1[1] * e2[2] - e1[2] * e2[1],
					e1[2] * e2[0] - e1[0] * e2[2],
					e1[0] * e2[1] - e1[1] * e2[0],
				};
				const V3 centroid = {
					v[i][0] + v[j][0] + v[k][0],
					v[i][1] + v[j][1] + v[k][1],
					v[i][2] + v[j][2] + v[k][2],
				};
				const double facing = normal[0] * centroid[0] +
				                      normal[1] * centroid[1] +
				                      normal[2] * centroid[2];

				// Против часовой стрелки снаружи <=> нормаль направлена наружу
				// Мы хотим, чтобы направление шло по часовой стрелке снаружи, поэтому нормаль должна быть направлена внутрь
				mesh.triangles.push_back(uint32_t(i));
				if (facing < 0.0) {
					mesh.triangles.push_back(uint32_t(j));
					mesh.triangles.push_back(uint32_t(k));
				} else {
					mesh.triangles.push_back(uint32_t(k));
					mesh.triangles.push_back(uint32_t(j));
				}
			}
		}
	}

	return mesh;
}

} // namespace geometry
