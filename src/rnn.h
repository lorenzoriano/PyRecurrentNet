/*
    <one line to give the program's name and a brief idea of what it does.>
    Copyright (C) <year>  <name of author>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

*/

#ifndef RNN_H
#define RNN_H

#include <boost/shared_array.hpp>
#include <exception>
#include <cmath>
#include <boost/math/special_functions/fpclassify.hpp>
#include <string>
#include <cstring>

#ifdef _MSC_VER
#include <float.h>  // for _isnan() on VC++
#define isnan(x) _isnan(x)  // VC++ uses _isnan() instead of isnan()
#define isinf(x) (!_finite(x) && !isnan(x))
typedef unsigned int uint;
#endif


struct rnn_value_exception : std::exception
{
    rnn_value_exception(std::string str) {
	msg = str;
    }

    const char* what() const throw() {
	return msg.c_str();
    }

    virtual ~rnn_value_exception() throw() {

    }

    private:
	std::string msg;
};

class RNN
{
    public:
		typedef double vt;
		typedef boost::shared_array<vt> vector;
		typedef boost::shared_array<vt> matrix;


		RNN(unsigned int hidden_size, unsigned int input_size, unsigned int output_size)
		{
			this->m_input_size = input_size;
			this->m_hidden_size = hidden_size;
			this->m_output_size = output_size;
			this->m_reserved_size = input_size + output_size;
			this->m_size = input_size + hidden_size + output_size;
			this->m_output_slice_left = this->m_input_size;
			this->m_output_slice_right = this->m_input_size + this->m_output_size;
			this->m_x.reset(new vt[this->m_size]);
			this->m_W.reset(new vt[this->m_size * this->m_size]);
			this->m_bias.reset(new vt[this->m_size]);

            zero_all();
		}

        void zero_all() {
            for (unsigned int i=0; i< this->m_size; i++) {
                access(m_x,i) = 0.0;
                access(m_bias,i) = 0.0;
                for (unsigned int j=0; j< this->m_size; j++) {
                    access(m_W,i,j) = 0.0;
                }
            }
        }

		vector operator ()(vt* input)
		{
			#define sigmoid(x) (1.0/(1+exp(-(x))))
			vector x_new = vector(new vt[this->m_size]);
	
			for (uint i = 0; i<this->m_input_size; i++) {
				if ( (input[i] > 1.0) || (input[i] < 0.0))
					throw rnn_value_exception("Input value is not between 0 and 1");

				if (isnan(input[i]) || isinf(input[i]))
					throw rnn_value_exception("Passing a non finite number!");

				access(m_x,i) = sigmoid(input[i]);
			}
	
			for (uint i = this->m_input_size; i<this->m_size; i++) {
				access(x_new,i) = access(m_bias,i);
				for (uint j = 0; j<this->m_size; j++) {
					access(x_new,i) += access(m_W,i,j) * access(m_x,j) ;
				}
				access(x_new,i) = sigmoid(access(x_new,i));
			}
	
			this->m_x = x_new;

			vector out = vector(new vt[this->m_output_size]);

			for (unsigned int i=0; i<this->m_output_size; i++){
				access(out,i) = access(m_x,i+m_output_slice_left);
			}
			return out;
		}

		vector call(vt* input) {
			return this->operator()(input);
		}

        vector evolve(vt* input, unsigned int steps ) {
            vector out;

            for (unsigned int i=0; i<steps; i++)
                out = call(input);
            return out;
        }
        

		void randomiseState() {
		    for (unsigned int i = 0; i < this->m_size; i++) {
			access(m_x,i) = rand() / vt(RAND_MAX);
		    }
		}

		vector get_x () const {
			return m_x;
		}
		vector get_bias() const {
			return m_bias;
		}
		matrix get_W() const {
			return m_W;
		}

		void set_x (const vector value) {
			m_x = value;
		}
		void set_bias(const vector value) {
			m_bias = value;
		}
		void set_W(const vector value) {
			m_W = value;
		}

		unsigned int hidden_size() const
		{
			return m_hidden_size;
		}

		unsigned int input_size() const
		{
			return m_input_size;
		}

		unsigned int output_size() const
		{
			return m_output_size;
		}

		unsigned int size() const
		{
			return m_size;
		}
  

    protected:

		vector m_x;
		matrix m_W;
		vector m_bias;

		unsigned int m_input_size;
		unsigned int m_hidden_size;
		unsigned int m_output_size;
		unsigned int m_size;
		unsigned int m_output_slice_left;
		unsigned int m_output_slice_right;
		unsigned int m_reserved_size;

		double& access(vector v, unsigned int i) {
				return v[i];
		}

		double access(vector v, unsigned int i) const {
			return v[i];
		}
	
		double& access(matrix m, unsigned int i, unsigned int j) {
			return m[i*this->m_size + j];
		}

		double access(matrix m, unsigned int i, unsigned int j) const {
			return m[i*this->m_size + j];
		}

};

#endif // RNN_H
