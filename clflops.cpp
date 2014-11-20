/******************************************************************************\
* Benchmarking tool to measure FLOPS on devices that support OpenCL.           *
* Copyright (C) 2015 Kenneth Hafdahl                                           *
*                                                                              *
* This program is free software; you can redistribute it and/or                *
* modify it under the terms of the GNU General Public License                  *
* as published by the Free Software Foundation; either version 2               *
* of the License, or (at your option) any later version.                       *
*                                                                              *
* This program is distributed in the hope that it will be useful,              *
* but WITHOUT ANY WARRANTY; without even the implied warranty of               *
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                *
* GNU General Public License for more details.                                 *
*                                                                              *
* You should have received a copy of the GNU General Public License            *
* along with this program; if not, write to the Free Software                  *
* Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301,   *
* USA.                                                                         *
\******************************************************************************/
#include <CL/cl.hpp>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <sys/time.h>
#include <unistd.h>
#include <vector>

using namespace std;

template<class T>
void initialize_data(T& vec, typename T::size_type size,
                     typename T::value_type min = 0,
                     typename T::value_type max = 1)
{
    static default_random_engine gen;
    uniform_real_distribution<typename T::value_type> dist(min, max);
    vec.resize(size);
    for (auto& elem : vec)
        elem = dist(gen);
}

void list_devices(vector<cl::Platform>& platforms, vector<cl::Device>& devices)
{
    auto it = platforms.begin();
    cl_platform_id id = 0;
    for (unsigned i = 0; i < devices.size(); ++i) {
        auto cur = devices.at(i).getInfo<CL_DEVICE_PLATFORM>();
        if (cur != id) {
            id = cur;
            cout << it++->getInfo<CL_PLATFORM_VENDOR>() << " "
                 << it->getInfo<CL_PLATFORM_NAME>() << ":" << endl;
        }
        cout << "[" << i << "] " <<
            devices.at(i).getInfo<CL_DEVICE_NAME>() << endl;
    }
}

void set_memory_test_size(const string& size, long unsigned& memory_test_size)
{
    stringstream ss(size);
    string prefix;
    ss >> memory_test_size;
    ss >> prefix;
    if (prefix == "M" || prefix == "m") {
        memory_test_size *= 1E6;
    } else if (prefix == "G" || prefix == "g") {
        memory_test_size *= 1E9;
    } else if (prefix.size()) {
        cerr << "Unidentified size prefix \""
             << prefix << "\"" << endl;
        exit(1);
    }
}

int main(int argc, char* argv[])
{
    vector<cl::Platform> platforms;
    vector<cl::Device> devices;
    unsigned long memory_test_size = 512E6;

    cl::Platform::get(&platforms);
    if (!platforms.size()) {
        cerr << "No platforms found. Verify runtime installation." << endl;
        exit(1);
    }

    for (auto& platform: platforms) {
        vector<cl::Device> tmp;
        platform.getDevices(CL_DEVICE_TYPE_ALL, &tmp);
        devices.insert(devices.end(), tmp.begin(), tmp.end());
    }

    bool list_devices_flag = false;
    int opt;
    while ((opt = getopt(argc, argv, ":ls:")) != -1) {
        switch (opt) {
        case 'l':
            if (list_devices_flag)
                break;
            list_devices_flag = true;
            list_devices(platforms, devices);
            break;
        case 's':
            set_memory_test_size(optarg, memory_test_size);
            break;
        default:
            cerr << "Unexpected case in getopt switch" << endl;
            exit(1);
        }
    }

    cl::Context context({devices[0]});
    cl::Program::Sources source;
    string code =
        "void kernel thread_add(global const float* A, global const float* B)"
        "{"
        "    (volatile float)A[get_global_id(0)] + B[get_global_id(0)];"
        "    (volatile float)A[get_global_id(0)] - B[get_global_id(0)];"
        "    (volatile float)A[get_global_id(0)] * B[get_global_id(0)];"
        "    (volatile float)A[get_global_id(0)] / B[get_global_id(0)];"
        "}";
    source.push_back({code.c_str(), code.length()});

    cl::Program program(context, source);
    if (program.build({devices[0]}) != CL_SUCCESS) {
        cerr << "Error building. Verify OpenCL installation." << endl;
        cout << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]) << endl;
        exit(1);
    }

    vector<float> avec, bvec, cvec;
    initialize_data(avec, memory_test_size / 3 / sizeof(float));
    initialize_data(bvec, memory_test_size / 3 / sizeof(float));
    cvec.resize(avec.size());

    cl::Buffer buff_A(context, CL_MEM_READ_ONLY, sizeof(float) * avec.size());
    cl::Buffer buff_B(context, CL_MEM_READ_ONLY, sizeof(float) * bvec.size());
    //cl::Buffer buff_C(context, CL_MEM_READ_WRITE, sizeof(float) * cvec.size());

    cl::CommandQueue queue(context, devices[0]);

    queue.enqueueWriteBuffer(buff_A, CL_TRUE, 0, sizeof(float) * avec.size(),
                             avec.data());
    queue.enqueueWriteBuffer(buff_B, CL_TRUE, 0, sizeof(float) * bvec.size(),
                             bvec.data());

    cl::Kernel thread_add(program, "thread_add");
    thread_add.setArg(0, buff_A);
    thread_add.setArg(1, buff_B);
    //thread_add.setArg(2, buff_C);
    cl::NDRange global(avec.size());
    cl::NDRange local(1);
    cl::Event event;

    struct timeval tv1, tv2;
    gettimeofday(&tv1, NULL);
    for (int i = 0; i < 50; ++i) {
        queue.enqueueNDRangeKernel(thread_add, cl::NullRange, global,
                                   local, NULL, &event);
        event.wait();
    }
    gettimeofday(&tv2, NULL);
    double time = tv2.tv_sec - tv1.tv_sec +
        (tv2.tv_usec - tv1.tv_usec) * 1.0E-6;
    cout << time << " s" << endl;
    cout << avec.size() * 4 * 50 / time / 1E9 << " GFLOPS" << endl;

    /*
    queue.enqueueReadBuffer(buff_C, CL_TRUE, 0, sizeof(float) * cvec.size(), cvec.data());

    // verify results are close to calculated results
    for (unsigned i = 0; i < cvec.size(); ++i) {
        if (abs(avec[i] / bvec[i] - cvec[i]) > cvec[i] * 1.0E-6) {
            cerr << "Incorrect OpenCL calculation!" << endl;
            cout << cvec[i] << " != " << avec[i] / bvec[i] << endl;
            exit(1);
        }
    }
    */
    return 0;
}
