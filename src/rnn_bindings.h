/*
 * rnn_bindings.h
 *
 *  Created on: 24 May 2011
 *      Author: pezzotto
 */

#ifndef RNN_BINDINGS_H_
#define RNN_BINDINGS_H_

#include <boost/python.hpp>
#include <arrayobject.h>
#include <boost/shared_array.hpp>
#include "rnn.h"

class RNN_Wrapper : public RNN {

public:

	RNN_Wrapper(unsigned int hidden_size, unsigned int input_size, unsigned int output_size) :
		RNN(hidden_size, input_size, output_size) {}

	PyObject* operator()(PyObject* o) {
		PyArrayObject* array = __array_from_object(o, m_input_size);

		double* data = (double*)PyArray_DATA(array);
		vector out = RNN::operator()(data);

		return __convert_from_data(out, m_output_size);
	}

	PyObject* call(PyObject* o) {
		return operator()(o);
	}

	PyObject* get_x() const {
		return __convert_from_data(RNN::get_x(), m_size);
	}

	PyObject* get_bias() const {
		return __convert_from_data(RNN::get_bias(), m_size);
	}

	PyObject* get_W() const {
		return __convert_from_data(RNN::get_W(), m_size, m_size);
	}

	void set_x(PyObject* o) {
		PyArrayObject* array = __array_from_object(o, m_size);
		double* array_data = (double*)PyArray_DATA(array);
		vector data = vector(new vt[m_size]);

		for (unsigned int i=0; i<m_size; i++) {
			data[i] = array_data[i];
		}

		RNN::set_x(data);
	}

	void set_bias(PyObject* o) {
		PyArrayObject* array = __array_from_object(o, m_size);
		double* array_data = (double*)PyArray_DATA(array);
		vector data = vector(new vt[m_size]);

		for (unsigned int i=0; i<m_size; i++) {
			data[i] = array_data[i];
		}

		RNN::set_bias(data);
	}

	void set_W(PyObject* o) {
		PyArrayObject* array = __matrix_from_object(o, m_size, m_size);
		double* array_data = (double*)PyArray_DATA(array);
		unsigned int d_size = m_size*m_size;
		vector data = matrix(new vt[d_size]);

		for (unsigned int i=0; i<d_size; i++) {
			data[i] = array_data[i];
		}

		RNN::set_W(vector(data));
	}

protected:
	PyObject* __convert_from_data(vector data, uint size) const{

		npy_intp dims[1];
		dims[0] = size;

		PyObject* out = PyArray_SimpleNew(1,dims, NPY_DOUBLE);
		
		for (unsigned int i=0; i<size; i++) {
			*((double*)PyArray_GETPTR1(out, i)) = data[i];
		}
		
		return out;
	}

	PyObject* __convert_from_data(matrix data, uint size1, uint size2) const{
		npy_intp dims[2];
		dims[0] = size1;
		dims[1] = size2;

		PyObject* out = PyArray_SimpleNew(2,dims, NPY_DOUBLE);

		for (unsigned int i=0; i<size1; i++) {
			for (unsigned int j=0; j<size1; j++) {
				*((double*)PyArray_GETPTR2(out, i,j)) = access(data,i,j);
			}
		}

		
		return out;
	}

	PyArrayObject* __array_from_object(PyObject* o, unsigned int size) const {
		PyArrayObject* array = (PyArrayObject*)PyArray_FROMANY(o,
						NPY_DOUBLE,1,1, NPY_CARRAY);

		//A few checks
		if (array == NULL)
			throw rnn_value_exception("Error during conversion. Wrong input?");
		if (array->nd > 1)
			throw rnn_value_exception("Wrong number of dimensions");
		if (uint(array->dimensions[0]) != size)
			throw rnn_value_exception("Wrong number of elements");

		return array;
	}

	PyArrayObject* __matrix_from_object(PyObject* o, unsigned int size1, unsigned int size2) const {
		PyArrayObject* array = (PyArrayObject*)PyArray_FROMANY(o,
						NPY_DOUBLE,2,2, NPY_CARRAY);

		//A few checks
		if (array == NULL)
			throw rnn_value_exception("Error during conversion. Wrong input?");
		if (array->nd > 2)
			throw rnn_value_exception("Wrong number of dimensions");
		if (uint(array->dimensions[0]) != size1)
			throw rnn_value_exception("Wrong number of rows");
		if (uint(array->dimensions[1]) != size2)
			throw rnn_value_exception("Wrong number of columns");

		return array;
	}

};

#endif /* RNN_BINDINGS_H_ */
