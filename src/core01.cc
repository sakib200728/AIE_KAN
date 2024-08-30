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

    float* in_ptr = in->ptr;  // Pointer to the input buffer
    float* target_ptr = target->ptr;  // Pointer to the target buffer
    float* out_ptr = out->ptr;  // Pointer to the output buffer
    float* grad_ptr = gradients->ptr;  // Pointer to the gradients buffer

    for (i = 0; i + vector_size <= num_knots && i + vector_size <= std::min(num_knots, 5); i += vector_size) {
        aie::vector<float, 8> input_vector = aie::load_v<8>(in_ptr);  // Load input vector
        in_ptr += vector_size;  // Increment the pointer by vector size

        aie::vector<float, 8> result_vector = aie::zeros<float, 8>();
        aie::vector<float, 8> error_vector;
        aie::vector<float, 8> grad_vector;

        for (int j = 0; j < 8; ++j) {
            if (i + j < num_knots) {
                // Fill the vector with the same value
                aie::vector<float, 8> coeff_vector = aie::generate<float, 8>(spline_coefficients_1[i + j]);
                aie::vector<float, 8> knot_vector = aie::generate<float, 8>(spline_knots_1[i + j]);

                result_vector += coeff_vector * (input_vector - knot_vector);
            }
        }

        // Pruning and sparsification step
        float pruning_threshold = 0.1f;  // Example threshold value for pruning
        result_vector = select(result_vector > pruning_threshold, result_vector, 0.0f);

        // Store the result vector using aie::store_v
        aie::store_v(out_ptr, result_vector);
        out_ptr += vector_size;  // Increment pointer by vector size

        error_vector = result_vector - aie::load_v<8>(target_ptr);  // Load target vector
        target_ptr += vector_size;  // Increment the pointer by vector size

        grad_vector = 2.0f * (error_vector + regularization_strength * result_vector);

        for (int j = 0; j < 8; ++j) {
            if (i + j < num_knots) {
                spline_coefficients_1[i + j] -= learning_rate * grad_vector[j];
            }
        }

        // Store the gradient vector
        aie::store_v(grad_ptr, grad_vector);
        grad_ptr += vector_size;  // Increment pointer by vector size
    }

    if (i < num_knots) {
        int remaining_elements = num_knots - i;
        aie::vector<float, 8> input_vector = aie::load_v<8>(in_ptr);  // Load input vector for remaining elements

        aie::vector<float, 8> result_vector = aie::zeros<float, 8>();
        aie::vector<float, 8> error_vector;
        aie::vector<float, 8> grad_vector;

        for (int j = 0; j < remaining_elements; ++j) {
            // Fill the vector with the same value
            aie::vector<float, 8> coeff_vector = aie::generate<float, 8>(spline_coefficients_1[i + j]);
            aie::vector<float, 8> knot_vector = aie::generate<float, 8>(spline_knots_1[i + j]);

            result_vector[j] += coeff_vector[j] * (input_vector[j] - knot_vector[j]);
        }

        // Pruning and sparsification step for remaining elements
        result_vector = select(result_vector > pruning_threshold, result_vector, 0.0f);

        // Store the result vector using aie::store_v
        aie::store_v(out_ptr, result_vector);
        out_ptr += remaining_elements;  // Increment pointer by remaining elements

        error_vector = result_vector - aie::load_v<8>(target_ptr);  // Load target vector for remaining elements
        target_ptr += remaining_elements;  // Increment pointer by remaining elements

        grad_vector = 2.0f * (error_vector + regularization_strength * result_vector);

        for (int j = 0; j < remaining_elements; ++j) {
            spline_coefficients_1[i + j] -= learning_rate * grad_vector[j];
        }

        // Store the gradient vector
        aie::store_v(grad_ptr, grad_vector);
        grad_ptr += remaining_elements;  // Increment pointer by remaining elements
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
