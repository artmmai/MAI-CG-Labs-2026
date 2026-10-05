#pragma once

#include <cmath>

namespace math {

constexpr float pi = 3.14159265358979323846f;

constexpr float radians(float degrees) {
	return degrees * (pi / 180.0f);
}

struct Mat4 {
	float e[16] = {};

	float& at(int row, int column) { return e[column * 4 + row]; }
	float at(int row, int column) const { return e[column * 4 + row]; }

	static Mat4 identity() {
		Mat4 m;
		m.at(0, 0) = m.at(1, 1) = m.at(2, 2) = m.at(3, 3) = 1.0f;
		return m;
	}
};

inline Mat4 operator*(const Mat4& a, const Mat4& b) {
	Mat4 r;
	for (int row = 0; row < 4; ++row) {
		for (int column = 0; column < 4; ++column) {
			float sum = 0.0f;
			for (int k = 0; k < 4; ++k) {
				sum += a.at(row, k) * b.at(k, column);
			}
			r.at(row, column) = sum;
		}
	}
	return r;
}

// Аффинные преобразования y = f(x) = Ax + b

inline Mat4 translation(float x, float y, float z) {
	Mat4 m = Mat4::identity();
	m.at(0, 3) = x;
	m.at(1, 3) = y;
	m.at(2, 3) = z;
	return m;
}

inline Mat4 scaling(float x, float y, float z) {
	Mat4 m = Mat4::identity();
	m.at(0, 0) = x;
	m.at(1, 1) = y;
	m.at(2, 2) = z;
	return m;
}

inline Mat4 rotationX(float angle) {
	const float c = std::cos(angle), s = std::sin(angle);
	Mat4 m = Mat4::identity();
	m.at(1, 1) = c;  m.at(1, 2) = -s;
	m.at(2, 1) = s;  m.at(2, 2) = c;
	return m;
}

inline Mat4 rotationY(float angle) {
	const float c = std::cos(angle), s = std::sin(angle);
	Mat4 m = Mat4::identity();
	m.at(0, 0) = c;   m.at(0, 2) = s;
	m.at(2, 0) = -s;  m.at(2, 2) = c;
	return m;
}

inline Mat4 rotationZ(float angle) {
	const float c = std::cos(angle), s = std::sin(angle);
	Mat4 m = Mat4::identity();
	m.at(0, 0) = c;  m.at(0, 1) = -s;
	m.at(1, 0) = s;  m.at(1, 1) = c;
	return m;
}

/* Ортогональная проекция прямоугольника [left,right] x [top,bottom] x [near,far]
 Так как ось Y Вулкана направлена вниз, "top" - это меньшее значение y, а
 "bottom" - большее. Это перевод, за которым следует масштабирование:
   | 2/(r-l)    0         0        -(r+l)/(r-l) |
   |   0      2/(b-t)     0        -(b+t)/(b-t) |
   |   0        0      1/(f-n)       -n/(f-n)   |
   |   0        0         0              1      |
*/
inline Mat4 orthographic(float left, float right, float top, float bottom,
                         float near_plane, float far_plane) {
	Mat4 m;
	m.at(0, 0) = 2.0f / (right - left);
	m.at(0, 3) = -(right + left) / (right - left);
	m.at(1, 1) = 2.0f / (bottom - top);
	m.at(1, 3) = -(bottom + top) / (bottom - top);
	m.at(2, 2) = 1.0f / (far_plane - near_plane);
	m.at(2, 3) = -near_plane / (far_plane - near_plane);
	m.at(3, 3) = 1.0f;
	return m;
}

/* Матрица перспективной проекции:
   | 1/((width / height)*tan(fov/2))      0           0        |
   |          0           1/tan(fov/2)    0           0        |
   |          0                 0      f/(f-n)   -f*n/(f-n)    |
   |          0                 0         1           0        |
*/
inline Mat4 perspective(float fov_y, float aspect, float near_plane, float far_plane) {
	const float t = std::tan(fov_y * 0.5f);
	Mat4 m;
	m.at(0, 0) = 1.0f / (aspect * t);
	m.at(1, 1) = 1.0f / t;
	m.at(2, 2) = far_plane / (far_plane - near_plane);
	m.at(2, 3) = -far_plane * near_plane / (far_plane - near_plane);
	m.at(3, 2) = 1.0f;
	return m;
}

} // namespace math
