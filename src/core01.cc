#include "core01.h"
#include "core01lut.h"

void kan_spline_kernel_core1(
    const int num_knots,
    input_window<float> *in,
    input_window<float> *target,
    output_window<float> *out,
    output_window<float> *gradients,
    float learning_rate,
    float regularization_strength
) {
    const int vector_size = 8;
    int i;

    for (i = 0; i + vector_size <= num_knots && i + vector_size <= std::min(num_knots, 5); i += vector_size) {
        aie::vector<float, vector_size> input_vector = window_readincr_v<vector_size>(in);
        aie::vector<float, vector_size> result_vector = aie::zeros<float, vector_size>();
        aie::vector<float, vector_size> error_vector;
        aie::vector<float, vector_size> grad_vector;

        for (int j = 0; j < vector_size; ++j) {
            if (i + j < num_knots) {
                aie::vector<float, vector_size> coeff_vector = aie::broadcast<float, vector_size>(spline_coefficients_1[i + j]);
                aie::vector<float, vector_size> knot_vector = aie::broadcast<float, vector_size>(spline_knots_1[i + j]);
                result_vector += coeff_vector * (input_vector - knot_vector);
            }
        }

        // Pruning and sparsification step
        float pruning_threshold = 0.1;  // Example threshold value for pruning
        result_vector = select(result_vector > pruning_threshold, result_vector, 0.0f);

        window_writeincr(out, result_vector);

        error_vector = result_vector - window_readincr_v<vector_size>(target);
        grad_vector = 2.0f * (error_vector + regularization_strength * result_vector);

        for (int j = 0; j < vector_size; ++j) {
            if (i + j < num_knots) {
                spline_coefficients_1[i + j] -= learning_rate * grad_vector[j];
            }
        }

        window_writeincr(gradients, grad_vector);
    }

    if (i < num_knots) {
        int remaining_elements = num_knots - i;
        aie::vector<float, 8> input_vector = window_readincr_v<8>(in);
        aie::vector<float, 8> result_vector = aie::zeros<float, 8>();
        aie::vector<float, 8> error_vector;
        aie::vector<float, 8> grad_vector;

        for (int j = 0; j < remaining_elements; ++j) {
            aie::vector<float, 8> coeff_vector = aie::broadcast<float, 8>(spline_coefficients_1[i + j]);
            aie::vector<float, 8> knot_vector = aie::broadcast<float, 8>(spline_knots_1[i + j]);
            result_vector[j] += coeff_vector[j] * (input_vector[j] - knot_vector[j]);
        }

        // Pruning and sparsification step for remaining elements
        result_vector = select(result_vector > pruning_threshold, result_vector, 0.0f);

        window_writeincr(out, result_vector);

        error_vector = result_vector - window_readincr_v<8>(target);
        grad_vector = 2.0f * (error_vector + regularization_strength * result_vector);

        for (int j = 0; j < remaining_elements; ++j) {
            spline_coefficients_1[i + j] -= learning_rate * grad_vector[j];
        }

        window_writeincr(gradients, grad_vector);
    }
}

void core01_top(
    input_window<float> &__restrict inA,
    input_window<float> &__restrict target,
    output_window<float> &__restrict out,
    output_window<float> &__restrict gradients,
    float learning_rate,
    float regularization_strength
) {
    const int num_knots = sizeof(spline_knots_1) / sizeof(spline_knots_1[0]);
    kan_spline_kernel_core1(num_knots, &inA, &target, &out, &gradients, learning_rate, regularization_strength);
}
