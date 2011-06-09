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

#include "rnn.h"
#include <boost/tuple/tuple.hpp>
#include "boost/tuple/tuple_io.hpp"
#include <boost/python.hpp>
#include <arrayobject.h>
#include <cmath>
#include <cstdlib>
#include <cassert>
#include <vector>
#include <iostream>
#include "rnn_bindings.h"

#define RAND() (float(rand())/float(RAND_MAX))
#define RAND_RANGE(a,b) ((b-a)*RAND() + a)
#define NORMALIZE(x,a,b) ((x-a)/(b-a))
#define DENORMALIZE(x,a,b) ( x*(b-a) + a )

class WorldState {

public:
	typedef boost::tuple<double, double, double> triplet;
	typedef boost::tuple<double, double> pair;

    WorldState()
    {
        robot_pos = boost::make_tuple(0,0,0);
        obj_pos = boost::make_tuple(2.5,2.5);
        table_dims = boost::make_tuple(1.4, 1.0);
        table_pos = boost::make_tuple(2.5, 2.5);
        max_x = 5.;
        max_y = 5.;
        min_x = 0;
        min_y = 0;
    }

    void move_robot_to(triplet pos)
    {
        if(!item_in_table(pos))
            robot_pos = pos;

    }
    bool item_in_table(pair obj) const
    {
        bool in_x = (obj.get<0>() < table_pos.get<0>() + table_dims.get<0>() / 2. && obj.get<0>() > table_pos.get<0>() - table_dims.get<0>() / 2.);
        bool in_y = (obj.get<1>() < table_pos.get<1>() + table_dims.get<1>() / 2. && obj.get<1>() > table_pos.get<1>() - table_dims.get<1>() / 2.);
        return in_x && in_y;
    }

    bool item_in_table(triplet obj) const
    {
        bool in_x = (obj.get<0>() < table_pos.get<0>() + table_dims.get<0>() / 2. && obj.get<0>() > table_pos.get<0>() - table_dims.get<0>() / 2.);
        bool in_y = (obj.get<1>() < table_pos.get<1>() + table_dims.get<1>() / 2. && obj.get<1>() > table_pos.get<1>() - table_dims.get<1>() / 2.);
        return in_x && in_y;
    }

    bool test_graspable() const
    {
        double dx = obj_pos.get<0>() - robot_pos.get<0>();
        double dy = obj_pos.get<1>() - robot_pos.get<1>();
        double angle = atan2(dy, dx) - robot_pos.get<2>();
        double dist = sqrt(dx * dx + dy * dy);
        return ( (angle > -M_PI_4) && (angle < M_PI_4) && (dist<1.0));
    }

    void random_robot_pos()
    {
        do {
			robot_pos = boost::make_tuple(RAND_RANGE(min_x, max_x),
					RAND_RANGE(min_y,max_y),
					RAND_RANGE(0., 2.0*M_PI));
		} while (item_in_table(robot_pos));
    }

    void random_object_pos()
    {
        double ux = table_pos.get<0>() + table_dims.get<0>() / 2.0;
        double lx = table_pos.get<0>() - table_dims.get<0>() / 2.0;
        double uy = table_pos.get<1>() + table_dims.get<1>() / 2.0;
        double ly = table_pos.get<1>() - table_dims.get<1>() / 2.0;
        obj_pos = boost::make_tuple(RAND_RANGE(lx,ux), RAND_RANGE(ly,uy));
        assert(item_in_table(obj_pos));
    }

    void random_table_pos() {
    	double lx = min_x + table_dims.get<0>();
    	double ux = max_x - table_dims.get<0>();

    	double ly = min_y + table_dims.get<1>();
    	double uy = max_y - table_dims.get<1>();

    	table_pos = boost::make_tuple(RAND_RANGE(lx, ux),
				RAND_RANGE(ly,uy));
    }

    void random_table_dims() {
    	table_dims = boost::make_tuple(RAND_RANGE(0.5, 1.4),
    								   RAND_RANGE(0.5, 1.4));
    }

    void randomize()
    {
    	if (short_version) {
    		random_robot_pos();
    		random_object_pos();
    	}
    	else {
			random_table_dims();
			random_table_pos();
			random_object_pos();
			random_robot_pos();
    	}
    }

    std::vector<double> normalised_input()
    {
    	if (short_version) {
    		std::vector<double> input(2);
			input[0] = NORMALIZE(table_pos.get<0>(),min_x,max_x);
			input[1] = NORMALIZE(table_pos.get<1>(),min_y,max_y);
			return input;
    	}
    	else {
			std::vector<double> input(6);
			input[0] = NORMALIZE(table_pos.get<0>(),min_x,max_x);
			input[1] = NORMALIZE(table_pos.get<1>(),min_y,max_y);
			input[2] = NORMALIZE(table_dims.get<0>(),min_x,max_x);
			input[3] = NORMALIZE(table_dims.get<1>(),min_y,max_y);
			input[4] = NORMALIZE(obj_pos.get<0>(),min_x,max_x);
			input[5] = NORMALIZE(obj_pos.get<1>(),min_y,max_y);
			return input;
    	}
    }

    template<class v_> std::vector<double> denormalize_pos(v_ vect, uint size) {
		std::vector<double> out(3);

		out[0] = DENORMALIZE(vect[0], min_x, max_x);
		out[1] = DENORMALIZE(vect[1], min_y, max_y);
		out[2] = DENORMALIZE(vect[2], 0, M_2_PI);
		return out;
	}

    pair getObj_pos() const
    {
        return obj_pos;
    }

    triplet getRobot_pos() const
    {
        return robot_pos;
    }

    void setObj_pos(pair obj_pos)
    {
        this->obj_pos = obj_pos;
    }

    void setRobot_pos(triplet robot_pos)
    {
        this->robot_pos = robot_pos;
    }

    pair getTable_dims() const
    {
        return table_dims;
    }

    pair getTable_pos() const
    {
        return table_pos;
    }

    void setTable_dims(pair table_dims)
    {
        this->table_dims = table_dims;
    }

    void setTable_pos(pair table_pos)
    {
        this->table_pos = table_pos;
    }

    void set_short(bool value) {
    	short_version = value;
    }

private:

	triplet robot_pos;
	pair obj_pos;
	pair table_dims;
	pair table_pos;
	double max_x;
	double max_y;
	double min_x;
	double min_y;
	bool short_version;

};

namespace bp = boost::python;

double test_statistics(unsigned int trials, bool short_version) {

	double success = 0;
	WorldState world;
	world.set_short(short_version);
	for (unsigned int i=0; i<trials; i++) {
		world.randomize();
		if (world.test_graspable()) {
			success++;
		}
	}
	return success/float(trials);
}


bool test_pos(bp::tuple robot_pos) {

	WorldState world;

	double x = bp::extract<double>(robot_pos[0]);
	double y = bp::extract<double>(robot_pos[1]);
	double th = bp::extract<double>(robot_pos[2]);

	WorldState::triplet world_r = boost::make_tuple<>(x,y,th);

	world.setRobot_pos(world_r);

	return world.test_graspable();
}

void print_world(const WorldState& world) {

	std::cout<<"Object pos:\t"<<world.getObj_pos()<<"\n";
	std::cout<<"Robot pos:\t"<<world.getRobot_pos()<<"\n";
	std::cout<<"Table pos:\t"<<world.getTable_pos()<<"\n";
	std::cout<<"Table dims:\t"<<world.getTable_dims()<<"\n";
}

void evaluate_out(RNN_Wrapper* wnet, bool short_version) {

	WorldState world;
	world.set_short(short_version);
	RNN* net = wnet;
	net->randomiseState();
	uint num_steps = 20;

	do {
		world.randomize();
	} while (world.test_graspable());

	RNN::vector netout;
	std::vector<double> input = world.normalised_input();

	for (unsigned int j=0; j<num_steps; j++) {
		netout = net->call(&input[0]);
	}

	std::vector<double> norm_out = world.denormalize_pos(netout, net->output_size());
	WorldState::triplet newpose = boost::make_tuple(norm_out[0], norm_out[1], norm_out[2]);
	world.move_robot_to(newpose);

	print_world(world);

	if (world.test_graspable())
		std::cout<<"Grasping ok\n";
	else
		std::cout<<"Not grasping\n";

}

double eval_func(RNN_Wrapper* wnet,
				 unsigned int num_trials,
				 unsigned int num_steps,
				 bool short_version) {

	WorldState world;
	world.set_short(short_version);
	unsigned int successes = 0;

	RNN* net = wnet;

	for (unsigned int i=0; i<num_trials; i++) {
		do {
			world.randomize();
		} while (world.test_graspable());
		net->randomiseState();

		std::vector<double> input = world.normalised_input();

		//stabilizing the network output
		RNN::vector netout;
		for (unsigned int j=0; j<num_steps; j++) {

			try {
				netout = net->call(&input[0]);
			}
			catch( rnn_value_exception& e) {
				std::cerr<<"Got an exception, here is the status\n";
				print_world(world);
				std::cerr<<"value: "<<netout[i]<<"\n";
				std::cerr<<"input value: ";
				for (unsigned int p=0; p<net->input_size(); p++)
					std::cerr<<input[i]<<" ";
				std::cerr<<"\n";

				throw e;
			}
		}

		std::vector<double> norm_out = world.denormalize_pos(netout, net->output_size());
		WorldState::triplet newpose = boost::make_tuple(norm_out[0], norm_out[1], norm_out[2]);
		world.move_robot_to(newpose);
		if (world.test_graspable())
			successes++;
	}
	double fitness = double(successes) / double(num_trials);
	return fitness;
}


BOOST_PYTHON_MODULE(libreaching){
	srand(time(NULL));
	import_array();

	bp::def("test_statistics", test_statistics);
	bp::def("test_pos", test_pos);
	bp::def("eval_func", eval_func);
	bp::def("evaluate_out", evaluate_out);
}



