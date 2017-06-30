
#include <stdio.h>
#include <stdlib.h>

/* MD5 algorithm from RFC 1321 */

#define MD5_STEP(a, b, expr, i, s, t)  \
do {                                   \
    a = (expr) + a + t + data[i];      \
    a = b + (a << s | a >> (32 - s));  \
} while (0)

#define ROUNDF(a,b,c,d,i,s,t)  MD5_STEP(a, b, b & c | d & ~b, i, s, t)
#define ROUNDG(a,b,c,d,i,s,t)  MD5_STEP(a, b, b & d | c & ~d, i, s, t)
#define ROUNDH(a,b,c,d,i,s,t)  MD5_STEP(a, b, b ^ c ^ d     , i, s, t)
#define ROUNDI(a,b,c,d,i,s,t)  MD5_STEP(a, b, c ^ (b | ~d)  , i, s, t)

__global__ void md5_hash_update (unsigned int * __restrict__ md5_state,
                                 const unsigned int * __restrict__ data)
{
    unsigned int a = md5_state[0];
    unsigned int b = md5_state[1];
    unsigned int c = md5_state[2];
    unsigned int d = md5_state[3];

    ROUNDF(a, b, c, d,  0,  7, 0xD76AA478);
    ROUNDF(d, a, b, c,  1, 12, 0xE8C7B756);
    ROUNDF(c, d, a, b,  2, 17, 0x242070DB);
    ROUNDF(b, c, d, a,  3, 22, 0xC1BDCEEE);
    ROUNDF(a, b, c, d,  4,  7, 0xF57C0FAF);
    ROUNDF(d, a, b, c,  5, 12, 0x4787C62A);
    ROUNDF(c, d, a, b,  6, 17, 0xA8304613);
    ROUNDF(b, c, d, a,  7, 22, 0xFD469501);
    ROUNDF(a, b, c, d,  8,  7, 0x698098D8);
    ROUNDF(d, a, b, c,  9, 12, 0x8B44F7AF);
    ROUNDF(c, d, a, b, 10, 17, 0xFFFF5BB1);
    ROUNDF(b, c, d, a, 11, 22, 0x895CD7BE);
    ROUNDF(a, b, c, d, 12,  7, 0x6B901122);
    ROUNDF(d, a, b, c, 13, 12, 0xFD987193);
    ROUNDF(c, d, a, b, 14, 17, 0xA679438E);
    ROUNDF(b, c, d, a, 15, 22, 0x49B40821);
    ROUNDG(a, b, c, d,  1,  5, 0xF61E2562);
    ROUNDG(d, a, b, c,  6,  9, 0xC040B340);
    ROUNDG(c, d, a, b, 11, 14, 0x265E5A51);
    ROUNDG(b, c, d, a,  0, 20, 0xE9B6C7AA);
    ROUNDG(a, b, c, d,  5,  5, 0xD62F105D);
    ROUNDG(d, a, b, c, 10,  9, 0x02441453);
    ROUNDG(c, d, a, b, 15, 14, 0xD8A1E681);
    ROUNDG(b, c, d, a,  4, 20, 0xE7D3FBC8);
    ROUNDG(a, b, c, d,  9,  5, 0x21E1CDE6);
    ROUNDG(d, a, b, c, 14,  9, 0xC33707D6);
    ROUNDG(c, d, a, b,  3, 14, 0xF4D50D87);
    ROUNDG(b, c, d, a,  8, 20, 0x455A14ED);
    ROUNDG(a, b, c, d, 13,  5, 0xA9E3E905);
    ROUNDG(d, a, b, c,  2,  9, 0xFCEFA3F8);
    ROUNDG(c, d, a, b,  7, 14, 0x676F02D9);
    ROUNDG(b, c, d, a, 12, 20, 0x8D2A4C8A);
    ROUNDH(a, b, c, d,  5,  4, 0xFFFA3942);
    ROUNDH(d, a, b, c,  8, 11, 0x8771F681);
    ROUNDH(c, d, a, b, 11, 16, 0x6D9D6122);
    ROUNDH(b, c, d, a, 14, 23, 0xFDE5380C);
    ROUNDH(a, b, c, d,  1,  4, 0xA4BEEA44);
    ROUNDH(d, a, b, c,  4, 11, 0x4BDECFA9);
    ROUNDH(c, d, a, b,  7, 16, 0xF6BB4B60);
    ROUNDH(b, c, d, a, 10, 23, 0xBEBFBC70);
    ROUNDH(a, b, c, d, 13,  4, 0x289B7EC6);
    ROUNDH(d, a, b, c,  0, 11, 0xEAA127FA);
    ROUNDH(c, d, a, b,  3, 16, 0xD4EF3085);
    ROUNDH(b, c, d, a,  6, 23, 0x04881D05);
    ROUNDH(a, b, c, d,  9,  4, 0xD9D4D039);
    ROUNDH(d, a, b, c, 12, 11, 0xE6DB99E5);
    ROUNDH(c, d, a, b, 15, 16, 0x1FA27CF8);
    ROUNDH(b, c, d, a,  2, 23, 0xC4AC5665);
    ROUNDI(a, b, c, d,  0,  6, 0xF4292244);
    ROUNDI(d, a, b, c,  7, 10, 0x432AFF97);
    ROUNDI(c, d, a, b, 14, 15, 0xAB9423A7);
    ROUNDI(b, c, d, a,  5, 21, 0xFC93A039);
    ROUNDI(a, b, c, d, 12,  6, 0x655B59C3);
    ROUNDI(d, a, b, c,  3, 10, 0x8F0CCC92);
    ROUNDI(c, d, a, b, 10, 15, 0xFFEFF47D);
    ROUNDI(b, c, d, a,  1, 21, 0x85845DD1);
    ROUNDI(a, b, c, d,  8,  6, 0x6FA87E4F);
    ROUNDI(d, a, b, c, 15, 10, 0xFE2CE6E0);
    ROUNDI(c, d, a, b,  6, 15, 0xA3014314);
    ROUNDI(b, c, d, a, 13, 21, 0x4E0811A1);
    ROUNDI(a, b, c, d,  4,  6, 0xF7537E82);
    ROUNDI(d, a, b, c, 11, 10, 0xBD3AF235);
    ROUNDI(c, d, a, b,  2, 15, 0x2AD7D2BB);
    ROUNDI(b, c, d, a,  9, 21, 0xEB86D391);

    md5_state[0] += a;
    md5_state[1] += b;
    md5_state[2] += c;
    md5_state[3] += d;
}

// Macro to catch CUDA errors in CUDA runtime calls
#define CUDA_SAFE_CALL(call)                                          \
do {                                                                  \
    cudaError_t err = call;                                           \
    if (cudaSuccess != err) {                                         \
        fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",\
                 __FILE__, __LINE__, cudaGetErrorString(err) );       \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
} while (0)

// Macro to catch CUDA errors in kernel launches
#define CHECK_LAUNCH_ERROR()                                          \
do {                                                                  \
    /* Check synchronous errors, i.e. pre-launch */                   \
    cudaError_t err = cudaGetLastError();                             \
    if (cudaSuccess != err) {                                         \
        fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",\
                 __FILE__, __LINE__, cudaGetErrorString(err) );       \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
    /* Check asynchronous errors, i.e. kernel failed (ULF) */         \
    err = cudaThreadSynchronize();                                    \
    if (cudaSuccess != err) {                                         \
        fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",\
                 __FILE__, __LINE__, cudaGetErrorString( err) );      \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
} while (0)

#if defined(_WIN32)
#if !defined(WIN32_LEAN_AND_MEAN)
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
static double second (void)
{
    LARGE_INTEGER t;
    static double oofreq;
    static int checkedForHighResTimer;
    static BOOL hasHighResTimer;

    if (!checkedForHighResTimer) {
        hasHighResTimer = QueryPerformanceFrequency (&t);
        oofreq = 1.0 / (double)t.QuadPart;
        checkedForHighResTimer = 1;
    }
    if (hasHighResTimer) {
        QueryPerformanceCounter (&t);
        return (double)t.QuadPart * oofreq;
    } else {
        return (double)GetTickCount() / 1000.0;
    }
}
#elif defined(__linux__) || defined(__APPLE__)
#include <stddef.h>
#include <sys/time.h>
static double second (void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}
#else
#error unsupported platform
#endif

int main (void)
{
    unsigned int block [16] = { 0x12345678 };
    unsigned int state [4] = { 0 };
    unsigned int *block_d;
    unsigned int *state_d;
    double startt, stopp, elapsed;
    dim3 gridDims = 65520;
    dim3 blockDims = 256;

    CUDA_SAFE_CALL (cudaMalloc ((void**)&block_d, sizeof(block)));
    CUDA_SAFE_CALL (cudaMalloc ((void**)&state_d, sizeof(state)));
    CUDA_SAFE_CALL (cudaMemcpy (block_d, block, sizeof(block), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL (cudaMemcpy (state_d, state, sizeof(state), cudaMemcpyHostToDevice));
    startt = second();
    startt = second();
    md5_hash_update<<<gridDims,blockDims>>>(state_d, block_d);
    CHECK_LAUNCH_ERROR();
    stopp = second();
    elapsed = stopp - startt;
    CUDA_SAFE_CALL (cudaMemcpy (state, state_d, sizeof(state), cudaMemcpyDeviceToHost));

    printf ("elapsed = %12.6f second %d hashes  %12.6f Mhashes/sec  %12.6f GB/sec\n",
            elapsed, gridDims.x * blockDims.x,
            (double)gridDims.x * blockDims.x / elapsed / 1e6,
            64.0*gridDims.x * blockDims.x / elapsed / 1e9);

    return EXIT_SUCCESS;
}


