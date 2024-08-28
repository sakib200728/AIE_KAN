#include "core06.h"
#include "core06lut.h"

void kan_spline_kernel_core6(
    const int num_knots,
    input_window<float> *in,
    input_window<float> *target,
    output_window<float> *out,
    output_window<float> *gradients,
    float learning_rate
) {
    // Vectorized input processing using AIE vector operations
    aie::vector<float, 8> input_vector;
    aie::vector<float, 8> result_vector;
    aie::vector<float, 8> error_vector;
    aie::vector<float, 8> grad_vector;

    for (int i = 0; i < num_knots; i += 8) {
        // Load input data into vector
        input_vector = window_readincr_v<8>(in);

        // Initialize result vector to zero
        result_vector = aie::zeros<float, 8>();

        // Apply the spline transformation using vectorized operations
        for (int j = 0; j < num_knots; ++j) {
            aie::vector<float, 8> coeff_vector = aie::broadcast<float, 8>(spline_coefficients_6[j]);
            aie::vector<float, 8> knot_vector = aie::broadcast<float, 8>(spline_knots_6[j]);

            result_vector += coeff_vector * (input_vector - knot_vector);
        }

        // Write the result to the output window
        window_writeincr(out, result_vector);

        // Load target data and calculate error
        aie::vector<float, 8> target_vector = window_readincr_v<8>(target);
        error_vector = result_vector - target_vector;

        // Compute gradients
        grad_vector = 2.0f * error_vector;

        // Update spline coefficients
        for (int j = 0; j < num_knots; ++j) {
            spline_coefficients_6[j] -= learning_rate * grad_vector[j];
        }

        // Output gradients for potential further processing
        window_writeincr(gradients, grad_vector);
    }
}

// Top-level function for Core 06
void core06_top(
    input_window<float> &__restrict inA,
    input_window<float> &__restrict target,
    output_window<float> &__restrict out,
    output_window<float> &__restrict gradients,
    float learning_rate
) {
    const int num_knots = sizeof(spline_knots_6) / sizeof(spline_knots_6[0]);
    kan_spline_kernel_core6(num_knots, &inA, &target, &out, &gradients, learning_rate);
}