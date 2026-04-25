#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../include/Model.hpp"
#include "../include/Layer.hpp"
#include "../include/Linear.hpp"
#include "../include/Conv2D.hpp"
#include "../include/MaxPool2D.hpp"
#include "../include/RNN.hpp"
#include "../include/Activations.hpp"
#include "../include/Loss.hpp"
#include "../include/Utils.hpp"
#include "../include/Embedding.hpp"
#include "../include/SelfAttention.hpp"
#include "../include/CrossAttention.hpp"

namespace py = pybind11;

PYBIND11_MODULE(fent, m) {
    py::class_<Layer>(m, "Layer")
        .def("forward", &Layer::forward)
        .def("backward", &Layer::backward);
    py::class_<Linear, Layer>(m, "Linear")
        .def(py::init<int, int>());
    py::class_<Conv2D, Layer>(m, "Conv2D")
        .def(py::init<int, int, int, int, int>());
    py::class_<RNN, Layer>(m, "RNN")
        .def(py::init<int, int>())
        .def("reset_states", &RNN::reset_states);
    py::class_<MaxPool2D, Layer>(m, "MaxPool2D")
        .def(py::init<int, int, int, int>());
    py::class_<Embedding, Layer>(m, "Embedding")
        .def(py::init<int, int>());
    py::class_<SelfAttention, Layer>(m, "SelfAttention")
        .def(py::init<int, int, int>());
    py::class_<CrossAttention, Layer>(m, "CrossAttention")
        .def(py::init<int, int, int, int>())
        .def("set_context", &CrossAttention::set_context)
        .def("get_context_gradient", &CrossAttention::get_context_gradient);
    py::class_<ReLU, Layer>(m, "ReLU")
        .def(py::init<>());
    py::class_<Tanh, Layer>(m, "Tanh")
        .def(py::init<>());

    py::class_<Model>(m, "Model")
        .def(py::init<>())
        .def("add", &Model::add, py::keep_alive<1, 2>())
        .def("forward", &Model::forward)
        .def("backward", &Model::backward)
        .def("save", &Model::save)
        .def("load", &Model::load);
        
    py::class_<Loss>(m, "Loss")
        .def_static("cross_entropy_loss", &Loss::cross_entropy_loss)
        .def_static("cross_entropy_gradient", &Loss::cross_entropy_gradient)
        .def_static("mse_loss", &Loss::mse_loss)
        .def_static("mse_gradient", &Loss::mse_gradient);

    auto futils = m.def_submodule("Utils");
    futils.def("softmax", &Utils::softmax);
    futils.def("sigmoid", &Utils::sigmoid);
}