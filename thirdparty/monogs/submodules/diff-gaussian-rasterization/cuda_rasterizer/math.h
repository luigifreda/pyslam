#pragma once
#include "helper_math.h"

struct mat33 {
    float3 cols[3];

  __host__ __device__ mat33() {}
  __host__ __device__ mat33(const float3 &c0, 
                            const float3 &c1,
                            const float3 &c2) {
    cols[0] = c0;
    cols[1] = c1;
    cols[2] = c2;
    }
    __host__ __device__ mat33(const float *data) {
        cols[0] = make_float3(data[0], data[1], data[2]);
        cols[1] = make_float3(data[3], data[4], data[5]);
        cols[2] = make_float3(data[6], data[7], data[8]);
    }

    __host__ __device__ static mat33 identity() {
        return mat33(make_float3(1, 0, 0),
                     make_float3(0, 1, 0),
                     make_float3(0, 0, 1));
    }

    __host__ __device__ static mat33 skew_symmetric(const float3 &v) {
        return mat33(make_float3(0, v.z, -v.y),
                     make_float3(-v.z, 0, v.x),
                     make_float3(v.y, -v.x, 0));
    }

    __host__ __device__ mat33 transpose() const {
        float3 c0 = cols[0];
        float3 c1 = cols[1];
        float3 c2 = cols[2];
        return mat33(make_float3(c0.x, c1.x, c2.x),
                     make_float3(c0.y, c1.y, c2.y),
                     make_float3(c0.z, c1.z, c2.z));
    }
    
    __host__ __device__ float &operator[](int i) { 
        float3 &col = cols[i / 3];
        return (&col.x)[i % 3]; 
    }    

    __host__ __device__ const mat33 operator+(const mat33 &m) const {
        float3 c0 = cols[0] + m.cols[0];
        float3 c1 = cols[1] + m.cols[1];
        float3 c2 = cols[2] + m.cols[2];
        return mat33(c0, c1, c2);
    }

    __host__ __device__ const mat33 operator*(const mat33 &m) const {
        float3 c0 = cols[0];
        float3 c1 = cols[1];
        float3 c2 = cols[2];
        float3 m0 = m.cols[0];
        float3 m1 = m.cols[1];
        float3 m2 = m.cols[2];

        float3 n0 = make_float3(c0.x * m0.x + c1.x * m0.y + c2.x * m0.z,
                                c0.y * m0.x + c1.y * m0.y + c2.y * m0.z,
                                c0.z * m0.x + c1.z * m0.y + c2.z * m0.z);
        float3 n1 = make_float3(c0.x * m1.x + c1.x * m1.y + c2.x * m1.z,
                                c0.y * m1.x + c1.y * m1.y + c2.y * m1.z,
                                c0.z * m1.x + c1.z * m1.y + c2.z * m1.z);
        float3 n2 = make_float3(c0.x * m2.x + c1.x * m2.y + c2.x * m2.z,
                                c0.y * m2.x + c1.y * m2.y + c2.y * m2.z,
                                c0.z * m2.x + c1.z * m2.y + c2.z * m2.z);   
        return mat33(n0, n1, n2);
    }

    __host__ __device__ const mat33 operator*(const float &s) const {
        float3 c0 = cols[0];
        float3 c1 = cols[1];
        float3 c2 = cols[2];
        return mat33(c0 * s, c1 * s, c2 * s);
    }

    __host__ __device__ const float3 operator*(const float3 &v) const {
        float3 c0 = cols[0];
        float3 c1 = cols[1];
        float3 c2 = cols[2];
        return make_float3(c0.x * v.x + c1.x * v.y + c2.x * v.z,
                           c0.y * v.x + c1.y * v.y + c2.y * v.z,
                           c0.z * v.x + c1.z * v.y + c2.z * v.z);
    }

    __host__ __device__ const mat33 operator-() const {
        float3 c0 = cols[0];
        float3 c1 = cols[1];
        float3 c2 = cols[2];
        return mat33(-c0, -c1, -c2);
    }
    
    friend __host__ __device__  mat33 operator*(const float &s, const mat33 &m) {
        return m * s;
    }
};



struct mat34 {
    float3 cols[4];
    __host__ __device__ mat34() {}
    __host__ __device__ mat34(const float3 &c0, 
          const float3 &c1,
          const float3 &c2,
          const float3 &c3) {
        cols[0] = c0;
        cols[1] = c1;
        cols[2] = c2;
        cols[3] = c3;
    }
    __host__ __device__ mat34(const float *data) {
        cols[0] = make_float3(data[0], data[1], data[2]);
        cols[1] = make_float3(data[3], data[4], data[5]);
        cols[2] = make_float3(data[6], data[7], data[8]);
        cols[3] = make_float3(data[9], data[10], data[11]);
    }
    __host__ __device__ mat34(const mat33 &m, const float3 &v) {
        cols[0] = m.cols[0];
        cols[1] = m.cols[1];
        cols[2] = m.cols[2];
        cols[3] = v;
    }

    __host__ __device__ float &operator[](int i) { 
        float3 &col = cols[i / 3];
        return (&col.x)[i % 3]; 
    }

    __host__ __device__ const mat34 operator+(const mat34 &m) const {
        float3 c0 = cols[0] + m.cols[0];
        float3 c1 = cols[1] + m.cols[1];
        float3 c2 = cols[2] + m.cols[2];
        float3 c3 = cols[3] + m.cols[3];
        return mat34(c0, c1, c2, c3);
    }
};

struct mat44 {
    float4 cols[4];
    __host__ __device__ mat44() {}
    __host__ __device__ mat44(const float4 &c0, const float4 &c1, const float4 &c2, const float4 &c3) {
        cols[0] = c0; cols[1] = c1; cols[2] = c2; cols[3] = c3;
    }
    __host__ __device__ mat44(const float *data) {
        cols[0] = make_float4(data[0], data[1], data[2], data[3]);
        cols[1] = make_float4(data[4], data[5], data[6], data[7]);
        cols[2] = make_float4(data[8], data[9], data[10], data[11]);
        cols[3] = make_float4(data[12], data[13], data[14], data[15]);
    }
    __host__ __device__ mat44(const mat33 &m, const float3 &v) {
        cols[0] = make_float4(m.cols[0], 0);
        cols[1] = make_float4(m.cols[1], 0);
        cols[2] = make_float4(m.cols[2], 0);
        cols[3] = make_float4(v, 1);
    }
    __host__ __device__ mat44(const mat34 &m) {
        cols[0] = make_float4(m.cols[0], 0);
        cols[1] = make_float4(m.cols[1], 0);
        cols[2] = make_float4(m.cols[2], 0);
        cols[3] = make_float4(m.cols[3], 1);
    }

    __host__ __device__ float &operator[](int i) { 
        float4 &col = cols[i / 4];
        return (&col.x)[i % 4]; 
    }

    __host__ __device__ mat44 operator+(const mat44 &m) const {
        float4 c0 = cols[0] + m.cols[0];
        float4 c1 = cols[1] + m.cols[1];
        float4 c2 = cols[2] + m.cols[2];
        float4 c3 = cols[3] + m.cols[3];
        return mat44(c0, c1, c2, c3);
    }

    __host__ __device__ mat44 operator*(const mat44 &m) const {
        float4 c0 = cols[0];
        float4 c1 = cols[1];
        float4 c2 = cols[2];
        float4 c3 = cols[3];
        float4 m0 = m.cols[0];
        float4 m1 = m.cols[1];
        float4 m2 = m.cols[2];
        float4 m3 = m.cols[3];

        float4 n0 = make_float4(c0.x * m0.x + c1.x * m0.y + c2.x * m0.z + c3.x * m0.w,
                                c0.y * m0.x + c1.y * m0.y + c2.y * m0.z + c3.y * m0.w,
                                c0.z * m0.x + c1.z * m0.y + c2.z * m0.z + c3.z * m0.w,
                                c0.w * m0.x + c1.w * m0.y + c2.w * m0.z + c3.w * m0.w);
        float4 n1 = make_float4(c0.x * m1.x + c1.x * m1.y + c2.x * m1.z + c3.x * m1.w,
                                c0.y * m1.x + c1.y * m1.y + c2.y * m1.z + c3.y * m1.w,
                                c0.z * m1.x + c1.z * m1.y + c2.z * m1.z + c3.z * m1.w,
                                c0.w * m1.x + c1.w * m1.y + c2.w * m1.z + c3.w * m1.w);
        float4 n2 = make_float4(c0.x * m2.x + c1.x * m2.y + c2.x * m2.z + c3.x * m2.w,
                                c0.y * m2.x + c1.y * m2.y + c2.y * m2.z + c3.y * m2.w,
                                c0.z * m2.x + c1.z * m2.y + c2.z * m2.z + c3.z * m2.w,
                                c0.w * m2.x + c1.w * m2.y + c2.w * m2.z + c3.w * m2.w);
        float4 n3 = make_float4(c0.x * m3.x + c1.x * m3.y + c2.x * m3.z + c3.x * m3.w,
                                c0.y * m3.x + c1.y * m3.y + c2.y * m3.z + c3.y * m3.w,
                                c0.z * m3.x + c1.z * m3.y + c2.z * m3.z + c3.z * m3.w,
                                c0.w * m3.x + c1.w * m3.y + c2.w * m3.z + c3.w * m3.w);
        return mat44(n0, n1, n2, n3);

    }

};

__forceinline__ __host__ __device__ float norm(const float3 &v) {
    return length(v);
}

struct SO3 {
    mat33 data_;
    __host__ __device__ SO3() {}
    __host__ __device__ SO3(const float3 &theta) {
        data_ = SO3::Exp(theta).data();
    }
    __host__ __device__ SO3(const mat33 &data) {
        data_ = data;
    }
    __host__ __device__  mat33 data() const {
        return data_;
    }

    __host__ __device__  mat33 static hat(const float3 &theta) {
        return mat33::skew_symmetric(theta);
    }

    __host__ __device__  SO3 static Exp(const float3 &theta) {
        mat33 W = SO3::hat(theta);
        mat33 W2 = W * W;
        float angle = norm(theta);
        mat33 I = mat33::identity();
        if (angle < 1e-5) {
            return SO3(I + W + 0.5f * W2);
        }
        else {
            return SO3(I + sin(angle) / angle * W + ((1 - cos(angle)) / (angle * angle)) * W2);
        }
    }
    __host__ __device__ float3 operator*(const float3 &v) const {
        return data_ * v;
    }

    __host__ __device__ SO3 operator*(const SO3 &R) const {
        return SO3(data_ * R.data_);
    }

    __host__ __device__ SO3 inverse() const {
        return SO3(data_.transpose());
    }
};

struct SE3 {
    SO3 R_data_;
    float3 t_data_;

    __host__ __device__ SE3() {}
    __host__ __device__ SE3(const float3 &rho, const float3 &theta) {
        SE3 T = SE3::Exp(rho, theta);
        t_data_ = T.t();
        R_data_ = T.R();
    }

    __host__ __device__ SE3(const float3 &t, const SO3 &R) {
        t_data_ = t;
        R_data_ = R;
    }

    __host__ __device__ SE3(const float *data) {
        mat44 T(data);
        t_data_ = make_float3(T.cols[3]);
        R_data_ = SO3(mat33(
            make_float3(T.cols[0]), make_float3(T.cols[1]), make_float3(T.cols[2]))
        );
    }

    __host__ __device__ SE3(const mat44 &data) {
        t_data_ = make_float3(data.cols[3]);
        R_data_ = SO3(mat33(
            make_float3(data.cols[0]), make_float3(data.cols[1]), make_float3(data.cols[2]))
        );
    }

    __host__ __device__ SO3 R() const {
        return R_data_;
    }

    __host__ __device__ float3 t() const {
        return t_data_;
    }

    __host__ __device__ mat44 data() const {
        return mat44(R_data_.data(), t_data_);
    }

    __host__ __device__ static mat44 hat(const float3 &rho, const float3 &theta) {
        mat33 W = SO3::hat(theta);
        mat44 T(W, rho);
        T.cols[3].w = 0;
        return T;
    }

    __host__ __device__ static SE3 Exp(const float3 &rho, const float3 &theta) {    
        mat33 W = SO3::hat(theta);
        mat33 W2 = W * W;
        SO3 R = SO3::Exp(theta);
        float angle = norm(theta);
        mat33 I = mat33::identity();
        mat33 V;
        if (angle < 1e-5) {
            V = I + 0.5f * W + 1.f / 6.f * W2;
        }
        else {
            V = I + W * ((1 - cos(angle)) / (angle * angle)) 
                + W2 * ((angle - sin(angle)) / (angle * angle * angle));
        }
        float3 t = V * rho;
        return SE3(t, R);
    }

    __host__ __device__ float3 operator*(const float3 &v) const {
        return R_data_ * v + t_data_;
    }

    __host__ __device__ SE3 operator*(const SE3 &T) const {
        return SE3(t_data_ + R_data_ * T.t_data_, R_data_ * T.R_data_);
    }

    __host__ __device__ SE3 inverse() const {
        SO3 R_inv = R_data_.inverse();
        float3 t = R_inv * t_data_;
        return SE3(-t, R_inv);
    }
};