//#include <ctime>
//#include <cstdlib>
//
//#include <iostream>
//#include <cstring>
#include "rnn_bindings.h"
#include <boost/python.hpp>

namespace bp = boost::python;


struct rnn_pickle_suite : bp::pickle_suite  {
    static boost::python::tuple getinitargs(const RNN_Wrapper& net) {
        return bp::make_tuple(net.hidden_size(),
			      net.input_size(),
			      net.output_size());
    }

    static boost::python::tuple getstate(boost::python::object net_obj) {
    	RNN_Wrapper const& net = bp::extract<RNN_Wrapper const&>(net_obj)();

    	bp::object o_W( bp::handle<>(bp::borrowed(net.get_W())) );
    	bp::object o_bias( bp::handle<>(bp::borrowed(net.get_bias())) );
    	bp::object o_x( bp::handle<>(bp::borrowed(net.get_x())) );

        return bp::make_tuple(net_obj.attr("__dict__"),
			      o_W,
			      o_bias,
			      o_x);
    }

    static void setstate(boost::python::object net_obj, boost::python::tuple state)
    {
    	RNN_Wrapper& net = bp::extract<RNN_Wrapper&>(net_obj)();

        if (bp::len(state) != 4)
        {
          PyErr_SetObject(PyExc_ValueError,
                          ("expected 2-item tuple in call to __setstate__; got %s"
                           % state).ptr()
              );
          bp::throw_error_already_set();
        }

        // restore the object's __dict__
        bp::dict d = bp::extract<bp::dict>(net_obj.attr("__dict__"))();
        d.update(state[0]);

        // restore the internal state of the C++ object


		net.set_W(bp::object(state[1]).ptr());
		net.set_bias(bp::object(state[2]).ptr());
		net.set_x(bp::object(state[3]).ptr());
    }

    static bool getstate_manages_dict() { return true; }
};

BOOST_PYTHON_MODULE(libcrnn){

    srand(time(NULL));
    import_array();
    
    bp::class_<RNN_Wrapper> ("RNN", bp::init<unsigned int, unsigned int, unsigned int>() )
	.def("__call__", &RNN_Wrapper::operator())
	.def("evolve", &RNN_Wrapper::evolve)
	.def("get_x", &RNN_Wrapper::get_x)
	.def("get_bias", &RNN_Wrapper::get_bias)
	.def("get_W", &RNN_Wrapper::get_W)
	.def("set_x", &RNN_Wrapper::set_x)
	.def("set_bias", &RNN_Wrapper::set_bias)
	.def("set_W", &RNN_Wrapper::set_W)
	.def("randomiseState", &RNN_Wrapper::randomiseState)
	.def("size", &RNN::size)
	.def("input_size", &RNN::input_size)
	.def("hidden_size", &RNN::hidden_size)
	.def("output_size", &RNN::output_size)
//	.def_readonly("output_slice_left", &RNN::m_output_slice_left)
//	.def_readonly("output_slice_right", &RNN::m_output_slice_right)
//	.def_readonly("reserved_size", &RNN::m_reserved_size)
	.def_pickle(rnn_pickle_suite())
	;
    
//    bp::type_id<PyArrayObject>()
//    boost::python::lvalue_from_pytype<extract_array_object,&PyArray_Type>();

}

