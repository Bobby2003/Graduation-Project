#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <vector>

namespace py = pybind11;

class VisualOdometry {
private:
    cv::Mat K;
    cv::Ptr<cv::ORB> orb;
    cv::Ptr<cv::BFMatcher> matcher;

    std::vector<cv::KeyPoint> prev_kp;
    cv::Mat prev_desc;
    cv::Mat prev_depth;Eigen::Matrix4f T_world;

public:
    VisualOdometry(float fx, float fy, float cx, float cy) {
        K = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
        orb = cv::ORB::create(800);
        matcher = cv::BFMatcher::create(cv::NORM_HAMMING, false);
        T_world = Eigen::Matrix4f::Identity();
    }

    py::array_t<float> track(py::array_t<uint16_t> depth_mm) {
        auto buf = depth_mm.request();
        cv::Mat depth(buf.shape[0], buf.shape[1], CV_16U, buf.ptr);

        // 深度 → 灰度
        cv::Mat gray, depth_f;
        depth.convertTo(depth_f, CV_32F);
        cv::normalize(depth_f, gray, 0, 255, cv::NORM_MINMAX, CV_8U);

        std::vector<cv::KeyPoint> kp;
        cv::Mat desc;
        orb->detectAndCompute(gray, cv::noArray(), kp, desc);

        if (prev_kp.empty() || desc.empty() || kp.size() < 8) {
            prev_kp = kp;
            prev_desc = desc.clone();
            prev_depth = depth.clone();
            return py::array_t<float>({4, 4}, T_world.data());
        }

        // 匹配
        std::vector<std::vector<cv::DMatch>> matches;
        matcher->knnMatch(prev_desc, desc, matches, 2);

        std::vector<cv::Point3f> pts3d;
        std::vector<cv::Point2f> pts2d;

        for (auto& m : matches) {
            if (m.size() == 2 && m[0].distance < 0.75f * m[1].distance) {
                cv::Point2f pt_prev = prev_kp[m[0].queryIdx].pt;
                int u = (int)pt_prev.x, v = (int)pt_prev.y;
                float z = prev_depth.at<uint16_t>(v, u) / 1000.0f;
                if (z > 0) {
                    float x = (u - K.at<double>(0, 2)) * z / K.at<double>(0, 0);
                    float y = (v - K.at<double>(1, 2)) * z / K.at<double>(1, 1);
                    pts3d.push_back(cv::Point3f(x, y, z));
                    pts2d.push_back(kp[m[0].trainIdx].pt);
                }
            }
        }

        if (pts3d.size() >= 6) {
            cv::Mat rvec, tvec, inliers;
            bool ok = cv::solvePnPRansac(pts3d, pts2d, K, cv::noArray(),
                                         rvec, tvec, false, 100, 2.0, 0.99, inliers);
            if (ok && inliers.rows >= 6) {
                cv::Mat R;
                cv::Rodrigues(rvec, R);
                Eigen::Matrix4f T_delta = Eigen::Matrix4f::Identity();
                for (int i = 0; i < 3; i++) {
                    for (int j = 0; j < 3; j++)
                        T_delta(i, j) = R.at<double>(i, j);
                    T_delta(i, 3) = tvec.at<double>(i);
                }
                T_world = T_world * T_delta.inverse();
            }
        }

        prev_kp = kp;
        prev_desc = desc.clone();
        prev_depth = depth.clone();

        return py::array_t<float>({4, 4}, T_world.data());
    }

    void reset() {
        prev_kp.clear();
        prev_desc.release();
        prev_depth.release();
        T_world = Eigen::Matrix4f::Identity();
    }
};

// 深度 → 世界坐标点云（批量）
py::tuple depth_to_world(py::array_t<uint16_t> depth_mm,py::array_t<uint8_t> vmask,
                         py::array_t<float> T_world_arr,
                         float fx, float fy, float cx, float cy) {
    auto d_buf = depth_mm.request();
    auto v_buf = vmask.request();
    auto T_buf = T_world_arr.request();

    int h = d_buf.shape[0], w = d_buf.shape[1];
    uint16_t* d_ptr = (uint16_t*)d_buf.ptr;
    uint8_t* v_ptr = (uint8_t*)v_buf.ptr;
    float* T_ptr = (float*)T_buf.ptr;

    Eigen::Matrix4f T;
    for (int i = 0; i < 16; i++) T.data()[i] = T_ptr[i];

    std::vector<Eigen::Vector3f> pts, col;

    for (int v = 0; v < h; v++) {
        for (int u = 0; u < w; u++) {
            int idx = v * w + u;
            if (d_ptr[idx] > 0 && v_ptr[idx] > 0) {
                float z = d_ptr[idx] / 1000.0f;
                float x = (u - cx) * z / fx;
                float y = -(v - cy) * z / fy;

                Eigen::Vector4f p_cam(x, y, z, 1.0f);
                Eigen::Vector4f p_w = T * p_cam;
                pts.push_back(p_w.head<3>());

                float t = std::min(z / 3.0f, 1.0f);
                col.push_back(Eigen::Vector3f(1.0f - t, 0, t));
            }
        }
    }

    py::array_t<float> pts_out({(int)pts.size(), 3});
    py::array_t<float> col_out({(int)col.size(), 3});
    auto pts_buf = pts_out.request();
    auto col_buf = col_out.request();
    float* pts_ptr = (float*)pts_buf.ptr;
    float* col_ptr = (float*)col_buf.ptr;

    for (size_t i = 0; i < pts.size(); i++) {
        pts_ptr[i * 3 + 0] = pts[i](0);
        pts_ptr[i * 3 + 1] = pts[i](1);
        pts_ptr[i * 3 + 2] = pts[i](2);
        col_ptr[i * 3 + 0] = col[i](0);
        col_ptr[i * 3 + 1] = col[i](1);
        col_ptr[i * 3 + 2] = col[i](2);
    }

    return py::make_tuple(pts_out, col_out);
}

PYBIND11_MODULE(slam_core, m) {
    py::class_<VisualOdometry>(m, "VisualOdometry")
        .def(py::init<float, float, float, float>())
        .def("track", &VisualOdometry::track)
        .def("reset", &VisualOdometry::reset);

    m.def("depth_to_world", &depth_to_world);
}