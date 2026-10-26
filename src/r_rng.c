/*
 *  R's default uniform random number generator, for use with standalone
 *  nmath (vendor/rmath) in place of nmath/standalone/sunif.c.
 *
 *  Copied, with the other RNG kinds removed, from R 4.5.3 src/main/RNG.c:
 *
 *  R : A Computer Language for Statistical Data Analysis
 *  Copyright (C) 1997--2025  The R Core Team
 *  Copyright (C) 1995, 1996  Robert Gentleman and Ross Ihaka
 *
 *  Mersenne-Twister code:
 *  Copyright (C) 1997, 1999 Makoto Matsumoto and Takuji Nishimura.
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, a copy is available at
 *  https://www.R-project.org/Licenses/
 *
 *  Matches R's defaults: RNGkind("Mersenne-Twister", "Inversion",
 *  "Rejection"). Normal deviates come from nmath's snorm.c, which uses
 *  Inversion in standalone builds.
 */

#include <math.h>
#include <stdint.h>

#include "r_rng.h"

typedef unsigned int Int32;

#define i2_32m1 2.328306437080797e-10 /* = 1/(2^32 - 1) */

/* dummy[0] is mti, dummy[1..624] is the state; same as .Random.seed[-1] */
static Int32 dummy[628]; // allow for optimizing compilers to read over bound

static double fixup(double x)
{
    /* ensure 0 and 1 are never returned */
    if(x <= 0.0) return 0.5*i2_32m1;
    if((1.0 - x) <= 0.0) return 1.0 - 0.5*i2_32m1;
    return x;
}

/* ===================  Mersenne Twister ========================== */

/* Period parameters */
#define N 624
#define M 397
#define MATRIX_A 0x9908b0df   /* constant vector a */
#define UPPER_MASK 0x80000000 /* most significant w-r bits */
#define LOWER_MASK 0x7fffffff /* least significant r bits */

/* Tempering parameters */
#define TEMPERING_MASK_B 0x9d2c5680
#define TEMPERING_MASK_C 0xefc60000
#define TEMPERING_SHIFT_U(y)  (y >> 11)
#define TEMPERING_SHIFT_S(y)  (y << 7)
#define TEMPERING_SHIFT_T(y)  (y << 15)
#define TEMPERING_SHIFT_L(y)  (y >> 18)

static Int32 *mt = dummy+1; /* the array for the state vector  */
static int mti=N+1; /* mti==N+1 means mt[N] is not initialized */

/* Initializing the array with a seed */
static void
MT_sgenrand(Int32 seed)
{
    int i;

    for (i = 0; i < N; i++) {
	mt[i] = seed & 0xffff0000;
	seed = 69069 * seed + 1;
	mt[i] |= (seed & 0xffff0000) >> 16;
	seed = 69069 * seed + 1;
    }
    mti = N;
}

static double MT_genrand(void)
{
    Int32 y;
    static Int32 mag01[2]={0x0, MATRIX_A};
    /* mag01[x] = x * MATRIX_A  for x=0,1 */

    mti = dummy[0];

    if (mti >= N) { /* generate N words at one time */
	int kk;

	if (mti == N+1)   /* if sgenrand() has not been called, */
	    MT_sgenrand(4357); /* a default initial seed is used   */

	for (kk = 0; kk < N - M; kk++) {
	    y = (mt[kk] & UPPER_MASK) | (mt[kk+1] & LOWER_MASK);
	    mt[kk] = mt[kk+M] ^ (y >> 1) ^ mag01[y & 0x1];
	}
	for (; kk < N - 1; kk++) {
	    y = (mt[kk] & UPPER_MASK) | (mt[kk+1] & LOWER_MASK);
	    mt[kk] = mt[kk+(M-N)] ^ (y >> 1) ^ mag01[y & 0x1];
	}
	y = (mt[N-1] & UPPER_MASK) | (mt[0] & LOWER_MASK);
	mt[N-1] = mt[M-1] ^ (y >> 1) ^ mag01[y & 0x1];

	mti = 0;
    }

    y = mt[mti++];
    y ^= TEMPERING_SHIFT_U(y);
    y ^= TEMPERING_SHIFT_S(y) & TEMPERING_MASK_B;
    y ^= TEMPERING_SHIFT_T(y) & TEMPERING_MASK_C;
    y ^= TEMPERING_SHIFT_L(y);
    dummy[0] = mti;

    return ( (double)y * 2.3283064365386963e-10 ); /* reals: [0,1)-interval */
}

/* ===================  R's entry points ========================== */

double unif_rand(void)
{
    return fixup(MT_genrand());
}

/* FixupSeeds(MERSENNE_TWISTER, initial); returns 0 if the state is all
   zeroes, where R would Randomize() instead */
static int FixupSeeds(int initial)
{
    int j, notallzero = 0;

    if(initial) dummy[0] = 624;
    /* No action unless user has corrupted .Random.seed */
    if(dummy[0] <= 0) dummy[0] = 624;
    /* check for all zeroes */
    for (j = 1; j <= 624; j++)
	if(dummy[j] != 0) {
	    notallzero = 1;
	    break;
	}
    return notallzero;
}

/* RNG_Init(MERSENNE_TWISTER, seed), as called by set.seed(seed) */
void castorbind_set_seed(int seed_)
{
    int j;
    Int32 seed = (Int32) seed_;

    /* Initial scrambling */
    for(j = 0; j < 50; j++)
	seed = (69069 * seed + 1);
    /* i_seed[0] is mti, *but* this is needed for historical consistency */
    for(j = 0; j < 1 + N; j++) {
	seed = (69069 * seed + 1);
	dummy[j] = seed;
    }
    FixupSeeds(1);
}

void castorbind_get_rng_state(int *state)
{
    for(int j = 0; j < CASTORBIND_RNG_STATE_LEN; j++)
	state[j] = (int) dummy[j];
}

int castorbind_set_rng_state(const int *state)
{
    Int32 saved[CASTORBIND_RNG_STATE_LEN];
    for(int j = 0; j < CASTORBIND_RNG_STATE_LEN; j++) {
	saved[j] = dummy[j];
	dummy[j] = (Int32) state[j];
    }
    if(!FixupSeeds(0)) {
	for(int j = 0; j < CASTORBIND_RNG_STATE_LEN; j++)
	    dummy[j] = saved[j];
	return -1;
    }
    return 0;
}

//generate a random non-negative integer < 2 ^ bits in 16 bit chunks
static double rbits(int bits)
{
    int_least64_t v = 0;
    for (int n = 0; n <= bits; n += 16) {
	int v1 = (int) floor(unif_rand() * 65536);
	v = 65536 * v + v1;
    }
    const int_least64_t one64 = 1L;
    // mask out the bits in the result that are not needed
    return (double) (v & ((one64 << bits) - 1));
}

/* R_unif_index with sample.kind = "Rejection" */
double R_unif_index(double dn)
{
    // rejection sampling from integers below the next larger power of two
    if (dn <= 0)
	return 0.0;
    int bits = (int) ceil(log2(dn));
    double dv;
    do { dv = rbits(bits); } while (dn <= dv);
    return dv;
}
