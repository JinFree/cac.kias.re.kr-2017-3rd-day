/* Generated by Cython 0.24 */

#ifndef __PYX_HAVE__pyend
#define __PYX_HAVE__pyend


#ifndef __PYX_HAVE_API__pyend

#ifndef __PYX_EXTERN_C
  #ifdef __cplusplus
    #define __PYX_EXTERN_C extern "C"
  #else
    #define __PYX_EXTERN_C extern
  #endif
#endif

#ifndef DL_IMPORT
  #define DL_IMPORT(_T) _T
#endif

__PYX_EXTERN_C DL_IMPORT(void) hello_world(void);

#endif /* !__PYX_HAVE_API__pyend */

#if PY_MAJOR_VERSION < 3
PyMODINIT_FUNC initpyend(void);
#else
PyMODINIT_FUNC PyInit_pyend(void);
#endif

#endif /* !__PYX_HAVE__pyend */
