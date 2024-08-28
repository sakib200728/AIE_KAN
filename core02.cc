#include "core02.h"
#include "core02lut.h"

void kan_spline_kernel_core2(
    const int num_knots,
    input_window<float> *in,
    input_window<float> *target,
    output_window<float> *out,
    output_window<float> *gradients,
    float learning_rate
) {
    // Vectorized input processing using AIE vector operations
    const int vector_size = 8;
    int i;

    // Process in chunks of vector_size (e.g., 8 elements at a time)
    for (i = 0; i + vector_size <= num_knots; i += vector_size) {
        aie::vector<float, vector_size> input_vector = window_readincr_v<vector_size>(in);
        aie::vector<float, vector_size> result_vector = aie::zeros<float, vector_size>();
        aie::vector<float, vector_size> error_vector;
        aie::vector<float, vector_size> grad_vector;

        for (int j = 0; j < vector_size; ++j) {
            aie::vector<float, vector_size> coeff_vector = aie::broadcast<float, vector_size>(spline_coefficients_2[i + j]);
            aie::vector<float, vector_size> knot_vector = aie::broadcast<float, vector_size>(spline_knots_2[i + j]);
            result_vector += coeff_vector * (input_vector - knot_vector);
        }

        window_writeincr(out, result_vector);

        // Calculate the error and gradients
        error_vector = result_vector - window_readincr_v<vector_size>(target);
        grad_vector = 2.0f * error_vector;

        // Update the spline coefficients
        for (int j = 0; j < vector_size; ++j) {
            spline_coefficients_2[i + j] -= learning_rate * grad_vector[j];
        }

        window_writeincr(gradients, grad_vector);
    }

    // Handle any remaining elements that don't fit into a full vector
    if (i < num_knots) {
        int remaining_elements = num_knots - i;
        aie::vector<float, vector_size> input_vector = window_readincr_v<remaining_elements>(in);
        aie::vector<float, vector_size> result_vector = aie::zeros<float, vector_size>();
        aie::vector<float, vector_size> error_vector;
        aie::vector<float, vector_size> grad_vector;

        for (int j = 0; j < remaining_elements; ++j) {
            aie::vector<float, vector_size> coeff_vector = aie::broadcast<float, vector_size>(spline_coefficients_2[i + j]);
            aie::vector<float, vector_size> knot_vector = aie::broadcast<float, vector_size>(spline_knots_2[i + j]);
            result_vector += coeff_vector * (input_vector - knot_vector);
        }

        window_writeincr(out, result_vector);

        // Calculate the error and gradients
        error_vector = result_vector - window_readincr_v<remaining_elements>(target);
        grad_vector = 2.0f * error_vector;

        // Update the spline coefficients
        for (int j = 0; j < remaining_elements; ++j) {
            spline_coefficients_2[i + j] -= learning_rate * grad_vector[j];
        }

        window_writeincr(gradients, grad_vector);
    }
}

// Top-level function for Core 02
void core02_top(
    input_window<float> &__restrict inA,
    input_window<float> &__restrict target,
    output_window<float> &__restrict out,
    output_window<float> &__restrict gradients,
    float learning_rate
) {
    const int num_knots = sizeof(spline_knots_2) / sizeof(spline_knots_2[0]);
    kan_spline_kernel_core2(num_knots, &inA, &target, &out, &gradients, learning_rate);
}
