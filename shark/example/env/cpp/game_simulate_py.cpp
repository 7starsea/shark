
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "game_simulate.h"

namespace py = pybind11;

class CatchBallSimulatePy : public CatchBallSimulate{

public:
    CatchBallSimulatePy(const py::tuple & screen, const py::tuple & ball, const py::tuple & ball_speed, const py::tuple & bar, 
					 double action_penalty=0, int waiting=0, bool is_continuous=false)
    : CatchBallSimulate(Size( int(py::int_(screen[1])), int(py::int_(screen[0])) ),
                        Size( int(py::int_(ball[1])), int(py::int_(ball[0])) ),
                        Size( int(py::int_(ball_speed[1])), int(py::int_(ball_speed[0]))),
                        Size( int(py::int_(bar[1])), int(py::int_(bar[0])) ),
                        action_penalty, waiting, is_continuous
                        ) {}

public:

    std::pair<bool, double> step_wrap(int action){
        double reward = 0;
        bool is_game_over  = step(action, reward);
        return std::pair<bool, double>( is_game_over, reward );
    }


    void get_display(py::array_t<uint8_t> & screen){
        if ( ! (screen.ndim() == 3 ) ) {
            throw std::runtime_error("Incorrect number of dimensions");
        }

        if( ! (screen.shape(0) == screen_size_.h  && screen.shape(1) == screen_size_.w  && screen.shape(2) == 3 ) ){
            throw std::runtime_error("Incorrect number of screen size");
        }
        
        uint8_3darray screen_t ( screen );
		draw(screen_t);        
    }
};


PYBIND11_MODULE(CatchBallSimulate, m)
{
     py::class_<CatchBallSimulatePy>(m, "CatchBallSimulate")
        .def(py::init<const py::tuple &, const py::tuple &, const py::tuple &, const py::tuple &, double, int, bool>(),
                py::arg("screen"), py::arg("ball"), py::arg("ball_speed"), py::arg("bar"), 
                py::arg("action_penalty")=0, py::arg("waiting")=0, py::arg("is_continuous")=false
            )

    .def("step", &CatchBallSimulatePy::step_wrap, "tuple(is_game_over, reward) step(action)")
    .def("reset", &CatchBallSimulate::reset)
    .def("seed", &CatchBallSimulate::seed)
    .def("reset_ball", &CatchBallSimulate::reset_ball)

    .def("get_display", &CatchBallSimulatePy::get_display)

    .def_property_readonly("screen_size",  & CatchBallSimulate::screen_size)
    .def_property_readonly("ball_size",  & CatchBallSimulate::ball_size )
    .def_property_readonly("bar_size",  & CatchBallSimulate::bar_size )
    .def_property("action_range", & CatchBallSimulate::action_range, & CatchBallSimulate::set_action_range)
    ;

}

















