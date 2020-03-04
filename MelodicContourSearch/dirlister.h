/*
 * dirlister.h
 *
 *  Created on: 2020/03/05
 *      Author: sin
 */

#ifndef DIRLISTER_H_
#define DIRLISTER_H_

#if defined(__linux__) || defined(__MACH__)
#include "dirlister_unix.h"
#elif defined(__WIN64)
#include "dirlister_win.h"
#endif


#endif /* DIRLISTER_H_ */
