#ifndef CORE01_H
#define CORE01_H

#include <stdint.h>
#include <adf.h>
#include "aie_api/aie.hpp"
#include "aie_api/aie_adf.hpp"
#include <aie_api/utils.hpp>

#ifndef INLINE
#ifndef INLINE_DECL
#define INLINE_DECL
#endif

#define __AIE_API_TYPES__HPP__

// Function declarations
void kan_spline_kernel_core1(
    const int num_knots,
    input_window<float> *in,
    input_window<float> *target,
    output_window<float> *out,
    output_window<float> *gradients,
    float learning_rate,
    float regularization_strength
);

void core01_top(
    input_window<float> &__restrict inA,
    input_window<float> &__restrict target,
    output_window<float> &__restrict out,
    output_window<float> &__restrict gradients,
    float learning_rate,
    float regularization_strength
);

#else
#  ifndef INLINE_DECL
#  ifdef __llvm__
#    define INLINE_DECL inline __attribute__((always_inline))
#  else
#    define INLINE_DECL inline
#  endif
#  endif
#  undef INLINE
#  include "core01.cc"
#  define INLINE
# endif

#endif // CORE01_H
