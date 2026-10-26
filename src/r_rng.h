#pragma once

/* R's Mersenne-Twister RNG state, laid out like .Random.seed[-1] */
#define CASTORBIND_RNG_STATE_LEN 625

#ifdef __cplusplus
extern "C" {
#endif

/* same stream as set.seed(seed) in R, with R's default RNGkind() */
void castorbind_set_seed(int seed);
void castorbind_get_rng_state(int *state);
/* returns -1, leaving the state unchanged, if state is invalid */
int castorbind_set_rng_state(const int *state);

#ifdef __cplusplus
}
#endif
