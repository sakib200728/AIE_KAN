#pragma once

#include <adf.h>
using namespace adf;

#include "core01.h"
#include "core02.h"
#include "core03.h"
#include "core04.h"
#include "core05.h"
#include "core06.h"
#include "core01lut.h"
#include "core02lut.h"
#include "core03lut.h"
#include "core04lut.h"
#include "core05lut.h"
#include "core06lut.h"

class myGraph : public graph {
private:
    // Kernel declarations
    kernel core01;
    kernel core02;
    kernel core03;
    kernel core04;
    kernel core05;
    kernel core06;

    // Declare LUT parameters
    parameter core01lut;
    parameter core02lut;
    parameter core03lut;
    parameter core04lut;
    parameter core05lut;
    parameter core06lut;

public:
    input_plio in[4];
    output_plio out[3];

    myGraph() {
        // Define input/output PLIOs
        in[0] = input_plio::create("prod_in1", plio_64_bits, "data/input01.txt");
        in[1] = input_plio::create("prod_in2", plio_64_bits, "data/input02.txt");
        in[2] = input_plio::create("prod_in3", plio_64_bits, "data/input03.txt");
        in[3] = input_plio::create("prod_in4", plio_64_bits, "data/input04.txt");

        out[0] = output_plio::create("prod_out1", plio_64_bits, "data/output01.txt");
        out[1] = output_plio::create("prod_out2", plio_64_bits, "data/output02.txt");
        out[2] = output_plio::create("prod_out3", plio_64_bits, "data/output03.txt");

        // Create kernels
        core01 = kernel::create(core01_top);
        core02 = kernel::create(core02_top);
        core03 = kernel::create(core03_top);
        core04 = kernel::create(core04_top);
        core05 = kernel::create(core05_top);
        core06 = kernel::create(core06_top);

        // Source files for the kernels
        source(core01) = "core01.cc";
        source(core02) = "core02.cc";
        source(core03) = "core03.cc";
        source(core04) = "core04.cc";
        source(core05) = "core05.cc";
        source(core06) = "core06.cc";

        // Set kernel locations
        location<kernel>(core01) = tile(8, 0);
        location<kernel>(core02) = tile(8, 1);
        location<kernel>(core03) = tile(8, 2);
        location<kernel>(core04) = tile(8, 3);
        location<kernel>(core05) = tile(8, 4);
        location<kernel>(core06) = tile(8, 5);

        // Define LUT parameters and connect them to kernels
        core01lut = parameter::array(spline_coefficients_1);
        core02lut = parameter::array(spline_coefficients_2);
        core03lut = parameter::array(spline_coefficients_3);
        core04lut = parameter::array(spline_coefficients_4);
        core05lut = parameter::array(spline_coefficients_5);
        core06lut = parameter::array(spline_coefficients_6);

        connect<>(core01lut, core01);
        connect<>(core02lut, core02);
        connect<>(core03lut, core03);
        connect<>(core04lut, core04);
        connect<>(core05lut, core05);
        connect<>(core06lut, core06);

        // Connect kernels to input/output PLIOs
        connect<>(in[0].out[0], core01.in[0]);
        connect<>(core01.out[0], core02.in[0]);
        connect<>(core02.out[0], out[0].in[0]);

        connect<>(in[1].out[0], core03.in[0]);
        connect<>(core03.out[0], core04.in[0]);
        connect<>(core04.out[0], out[1].in[0]);

        connect<>(in[2].out[0], core05.in[0]);
        connect<>(core05.out[0], core06.in[0]);
        connect<>(core06.out[0], out[2].in[0]);

        // Single buffer management
        single_buffer(core01.in[0]);
        single_buffer(core02.in[0]);
        single_buffer(core03.in[0]);
        single_buffer(core04.in[0]);
        single_buffer(core05.in[0]);
        single_buffer(core06.in[0]);

        single_buffer(core01.out[0]);
        single_buffer(core02.out[0]);
        single_buffer(core03.out[0]);
        single_buffer(core04.out[0]);
        single_buffer(core05.out[0]);
        single_buffer(core06.out[0]);

        // Runtime ratios for each kernel
        runtime<ratio>(core01) = 0.6;
        runtime<ratio>(core02) = 0.6;
        runtime<ratio>(core03) = 0.6;
        runtime<ratio>(core04) = 0.6;
        runtime<ratio>(core05) = 0.6;
        runtime<ratio>(core06) = 0.6;
    }
};
