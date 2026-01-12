#include "stickyvo_lgs/lgs_frontend.hpp"
#include <stdexcept>
#include <string>
#include <mutex>
#include <iostream>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace stickyvo_lgs {

static py::dtype dtype_uint8() { return py::dtype::of<uint8_t>(); }

// --- Interpreter Management ---

struct LgsFrontend::Impl {
  LgsConfig cfg;
  py::object fn;

  static std::mutex& interp_mutex() {
    static std::mutex m;
    return m;
  }
  static int& interp_refcount() {
    static int c = 0;
    return c;
  }
  
  // Managed release: Ensures the GIL acquired by py::initialize_interpreter()
  // is eventually released after initial imports are complete.
  static std::unique_ptr<py::gil_scoped_release>& gil_release() {
    static std::unique_ptr<py::gil_scoped_release> release;
    return release;
  }

  explicit Impl(const LgsConfig& c) : cfg(c) {
    std::lock_guard<std::mutex> lk(interp_mutex());
    if (interp_refcount() == 0) py::initialize_interpreter();
    interp_refcount()++;

    try {
      py::module mod = py::module::import(cfg.python_module.c_str());
      fn = mod.attr(cfg.python_func.c_str());
      
      if (!gil_release()) {
        gil_release() = std::make_unique<py::gil_scoped_release>();
      }
    } catch (const std::exception& e) {
      throw std::runtime_error(std::string("LGS Python import failed: ") + e.what());
    }
  }

  ~Impl() {
    {
      py::gil_scoped_acquire gil;
      try {
        py::module mod = py::module::import(cfg.python_module.c_str());
        if (py::hasattr(mod, "shutdown_lgs")) mod.attr("shutdown_lgs")();
      } catch (...) {}
      fn = py::none();
    }
    std::lock_guard<std::mutex> lk(interp_mutex());
    interp_refcount()--;
    if (interp_refcount() == 0) {
      gil_release().reset(); 
      py::finalize_interpreter();
    }
  }
};

// --- Array Conversion ---

static py::array imageview_to_numpy(const ImageView& img) {
  if (!img.data || img.width <= 0 || img.height <= 0 || img.stride_bytes <= 0) {
    throw std::runtime_error("Invalid ImageView");
  }

  if (img.format == ImageView::Format::kGray8) {
    return py::array(dtype_uint8(), {img.height, img.width}, {img.stride_bytes, 1}, img.data, py::none());
  }

  return py::array(dtype_uint8(), {img.height, img.width, 3}, {img.stride_bytes, 3, 1}, img.data, py::none());
}

// --- Parse Helpers ---

static std::vector<Vec2f> parse_kpts(const py::handle& arr_h) {
  py::array arr = py::reinterpret_borrow<py::array>(arr_h);
  if (arr.ndim() != 2 || arr.shape(1) != 2) throw std::runtime_error("kpts must be Nx2");

  auto r = arr.cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
  auto buf = r.unchecked<2>();
  std::vector<Vec2f> out; out.reserve((size_t)buf.shape(0));
  for (ssize_t i=0; i<buf.shape(0); ++i) out.push_back({buf(i,0), buf(i,1)});
  return out;
}

static std::vector<Match> parse_matches(const py::handle& arr_h) {
  py::array arr = py::reinterpret_borrow<py::array>(arr_h);
  if (arr.ndim() != 2 || arr.shape(1) != 2) throw std::runtime_error("matches must be Mx2");

  auto r = arr.cast<py::array_t<int, py::array::c_style | py::array::forcecast>>();
  auto buf = r.unchecked<2>();
  std::vector<Match> out; out.reserve((size_t)buf.shape(0));
  for (ssize_t i=0; i<buf.shape(0); ++i) out.push_back({buf(i,0), buf(i,1)});
  return out;
}

static std::vector<Line2D> parse_lines(const py::handle& arr_h) {
  py::array arr = py::reinterpret_borrow<py::array>(arr_h);
  if (arr.ndim() == 2 && arr.shape(1) == 4) {
    auto r = arr.cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
    auto buf = r.unchecked<2>();
    std::vector<Line2D> out; out.reserve((size_t)buf.shape(0));
    for (ssize_t i=0; i<buf.shape(0); ++i) out.push_back({Vec2f{buf(i,0), buf(i,1)}, Vec2f{buf(i,2), buf(i,3)}});
    return out;
  }
  if (arr.ndim() == 3 && arr.shape(1) == 2 && arr.shape(2) == 2) {
    auto r = arr.cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
    auto buf = r.unchecked<3>();
    std::vector<Line2D> out; out.reserve((size_t)buf.shape(0));
    for (ssize_t i=0; i<buf.shape(0); ++i) out.push_back({Vec2f{buf(i,0,0), buf(i,0,1)}, Vec2f{buf(i,1,0), buf(i,1,1)}});
    return out;
  }
  throw std::runtime_error("lines must be Lx4 or Lx2x2");
}

static py::handle dict_get_any(const py::dict& d, std::initializer_list<const char*> keys) {
  for (const char* k : keys) if (d.contains(k)) return d[k];
  return py::handle();
}

// --- Frontend Entry ---

LgsFrontend::LgsFrontend(const LgsConfig& cfg) : impl_(std::make_unique<Impl>(cfg)) {}
LgsFrontend::~LgsFrontend() = default;

PairMatches LgsFrontend::infer_pair(const ImageView& img0, const ImageView& img1, const CameraIntrinsics& K) const {
  py::gil_scoped_acquire gil;
  py::array np0 = imageview_to_numpy(img0), np1 = imageview_to_numpy(img1);
  py::dict Kd; Kd["fx"] = K.fx; Kd["fy"] = K.fy; Kd["cx"] = K.cx; Kd["cy"] = K.cy;
  py::dict cfgd; cfgd["model_dir"] = impl_->cfg.model_dir; cfgd["max_keypoints"] = impl_->cfg.max_keypoints; cfgd["use_gpu"] = impl_->cfg.use_gpu;

  py::object ret = impl_->fn(np0, np1, Kd, cfgd);
  py::dict d = ret.cast<py::dict>();
  PairMatches out;

  if (!d.contains("kpts0") || !d.contains("kpts1")) throw std::runtime_error("Missing kpts0/kpts1");
  out.f0.keypoints = parse_kpts(d["kpts0"]); out.f1.keypoints = parse_kpts(d["kpts1"]);

  py::handle pm = dict_get_any(d, {"pt_matches", "matches_points"});
  if (!pm) throw std::runtime_error("Missing point matches");
  out.point_matches = parse_matches(pm);

  py::handle lm = dict_get_any(d, {"line_matches", "matches_lines"});
  if (d.contains("lines0") && d.contains("lines1") && lm) {
    out.f0.lines = parse_lines(d["lines0"]);
    out.f1.lines = parse_lines(d["lines1"]);
    out.line_matches = parse_matches(lm);
  }

  if (d.contains("num_inliers_points")) out.num_inliers_points = d["num_inliers_points"].cast<int>();
  if (d.contains("num_inliers_lines"))  out.num_inliers_lines  = d["num_inliers_lines"].cast<int>();
  if (d.contains("score"))              out.score              = d["score"].cast<double>();

  return out;
}

void LgsFrontend::reset_state() {
  try {
    py::gil_scoped_acquire gil;
    py::module mod = py::module::import(impl_->cfg.python_module.c_str());
    if (py::hasattr(mod, "reset_lgs_sequence")) mod.attr("reset_lgs_sequence")();
  } catch (...) {}
}

} // namespace stickyvo_lgs
