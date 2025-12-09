#include <stdio.h>

#define GLOBAL_TOL 1e-5f

#define RED   "\x1b[31m"
#define GRN   "\x1b[32m"
#define RESET "\x1b[0m"

#define TEST_START(name) \
    do { \
        printf( "=== Testing: %s (%s:%d) ===\n", (name), __FILE__, __LINE__); \
    } while (0)

#define TEST_END(name) \
    do { \
        printf(GRN "=== PASSED: %s ===\n\n" RESET, (name)); \
    } while (0)

#define UNREACHABLE(msg) \
    do { \
        printf( "[FAIL] UNREACHABLE at %s:%d: %s\n", \
                __FILE__, __LINE__, msg); \
        return 1; \
    } while (0)

#define ASSERT_TRUE(name, cond) \
    do { \
        if (!(cond)) { \
            printf( \
                RED "[FAIL] %s (%s:%d): condition failed\n" RESET, \
                (name), __FILE__, __LINE__); \
            return 1; \
        } \
        printf(GRN "[OK]   %s (%s:%d)\n" RESET, (name), __FILE__, __LINE__); \
    } while (0)

#define ASSERT_NOT_NULL(name, ptr) \
    do { \
        if ((ptr) == NULL) { \
            printf( \
                RED "[FAIL] %s (%s:%d): pointer is NULL\n" RESET, \
                (name), __FILE__, __LINE__); \
            return 1; \
        } \
        printf(GRN "[OK]   %s (%s:%d)\n" RESET, (name), __FILE__, __LINE__); \
    } while (0)

#define ASSERT_FLOAT_EQ(name, a, b, tol)                                      \
    do {                                                                      \
        float _da = (a);                                                      \
        float _db = (b);                                                      \
        float _d  = fabsf(_da - _db);                                         \
        if (_d > (tol)) {                                                     \
            printf(                                                   \
                RED "[FAIL] %s (%s:%d): got %.6f expected %.6f (|diff|=%.6f)\n" RESET, \
                (name), __FILE__, __LINE__, _da, _db, _d);                    \
            return 1;                                                         \
        }                                                                     \
        printf( GRN "[OK]   %s (%s:%d)\n" RESET, (name), __FILE__, __LINE__); \
    } while (0)

#define ASSERT_FLOAT_EQ_ARR(name, a, b, n, tol)                               \
    do {                                                                      \
        int _fail = 0;                                                        \
        for (int _i = 0; _i < (n); _i++) {                                    \
            float _da = (a)[_i];                                              \
            float _db = (b)[_i];                                              \
            float _d  = fabsf(_da - _db);                                     \
            if (_d > (tol)) {                                                 \
                printf(                                               \
                    RED "[FAIL] %s[%d] (%s:%d): got %.6f expected %.6f (|diff|=%.6f)\n" RESET, \
                    (name), _i, __FILE__, __LINE__, _da, _db, _d);            \
                _fail = 1;                                                    \
            }                                                                 \
        }                                                                     \
        if (_fail) return 1;                                                  \
        printf( GRN "[OK]   %s (%s:%d)\n" RESET, (name), __FILE__, __LINE__); \
    } while (0)
