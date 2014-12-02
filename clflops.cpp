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
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <fstream>
#include <sstream>
#include <streambuf>
#include <sys/time.h>
#include <unistd.h>
#include <vector>

using namespace std;

const string CL_FILE_NAME("vectorops.cl");

// Initialize vec with size random values between min and max.
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


// Verify that the OpenCL device computes the square root correctly.
template<class T>
bool verify_data(T& data, typename T::value_type min = 0,
                 typename T::value_type max = 1)
{
    default_random_engine gen;
    uniform_real_distribution<typename T::value_type> dist(min, max);
    for (auto& elem : data) {
        if (abs(elem * elem - dist(gen)) > 1.0E-6) {
            return false;
        }
    }
    return true;
}

// Print all available OpenCL devices.
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

// Parse command line input for memory size.
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


// Compile and run benchmarking functions on OpenCL device.
void run_vector_ops(cl::Device& device, const string& code, vector<float>& data)
{
    cout << device.getInfo<CL_DEVICE_NAME>() << endl;
    cl::Context context({device});
    cl::Program::Sources source;
    source.push_back({code.c_str(), code.length()});

    cl::Program program(context, source);
    if (program.build({device}) != CL_SUCCESS) {
        cerr << "Error building. Verify OpenCL installation." << endl;
        cout << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << endl;
        exit(1);
    }

    cl::Buffer buff(context, CL_MEM_READ_WRITE, sizeof(float) * data.size());

    cl::CommandQueue queue(context, device);

    queue.enqueueWriteBuffer(buff, CL_TRUE, 0, sizeof(float) * data.size(),
                             data.data());

    // Run range based benchmark
    cout << setw(15) << left << "Range Based:";
    cl::Kernel range_op(program, "range_op");
    range_op.setArg(0, buff);
    int size = data.size();
    range_op.setArg(1, size);

    size_t nthreads = device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
    cl::NDRange global(nthreads);
    cl::NDRange local(1);
    cl::Event event;

    struct timeval tv1, tv2;
    gettimeofday(&tv1, NULL);
    queue.enqueueNDRangeKernel(range_op, cl::NullRange,
                               global, local, NULL, &event);
    event.wait();
    gettimeofday(&tv2, NULL);
    double time = tv2.tv_sec - tv1.tv_sec +
        (tv2.tv_usec - tv1.tv_usec) * 1.0E-6;

    vector<float> verify(data.size() / 100);
    queue.enqueueReadBuffer(buff, CL_TRUE, 0, sizeof(float) * verify.size(),
                            verify.data());
    if (!verify_data(verify)) {
        cerr << "Invalid computation from device." << endl;
        return;
    }

    cout << data.size() / time / 1.0E6 << "M Elements Per Second" << endl;

    queue.enqueueWriteBuffer(buff, CL_TRUE, 0, sizeof(float) * data.size(),
                             data.data());

    // Run element based benchmark
    cout << setw(15) << "Element Based:";
    cl::Kernel element_op(program, "element_op");
    element_op.setArg(0, buff);

    global = cl::NDRange(data.size());

    gettimeofday(&tv1, NULL);
    queue.enqueueNDRangeKernel(element_op, cl::NullRange,
                               global, local, NULL,
                               &event);
    event.wait();
    gettimeofday(&tv2, NULL);
    time = tv2.tv_sec - tv1.tv_sec +
           (tv2.tv_usec - tv1.tv_usec) * 1.0E-6;

    queue.enqueueReadBuffer(buff, CL_TRUE, 0, sizeof(float) * verify.size(),
                            verify.data());
    if (!verify_data(verify)) {
        cerr << "Invalid computation from device." << endl;
        return;
    }

    cout << data.size() / time / 1.0E6 << "M Elements Per Second" << endl;
    cout << endl;
}

int main(int argc, char* argv[])
{
    vector<cl::Platform> platforms;
    vector<cl::Device> devices;
    unsigned long memory_test_size = 512E6;
    bool device_index_set = false;
    unsigned device_index = 0;

    // Find available OpenCL devices
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

    // Parse command line options
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

    if (list_devices_flag)
        exit(0);

    if (optind < argc) {
        device_index_set = true;
        stringstream(argv[optind]) >> device_index;
        if (device_index >= devices.size()) {
            cerr << "No device " << device_index << " found." << endl;
            exit(1);
        }
    }

    // Read CL source file
    ifstream source_file(CL_FILE_NAME);
    if (!source_file.is_open()) {
        cerr << "Error opening " << CL_FILE_NAME << " for reading "<< endl;
        exit(1);
    }
    string code((istreambuf_iterator<char>(source_file)),
                istreambuf_iterator<char>());

    vector<float> data;
    initialize_data(data, memory_test_size / sizeof(float));

    if (device_index_set) {
        run_vector_ops(devices[device_index], code, data);
    } else {
        for (auto& device : devices)
            run_vector_ops(device, code, data);
    }

    return 0;
}
