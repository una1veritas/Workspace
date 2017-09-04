/*
 *  String.cpp
 *  Demo
 *
 *  Created by ‰º‰’ ^ˆê on 10/05/23.
 *  Copyright 2010 ‹ãBH‹Æ‘åŠwî•ñHŠw•”. All rights reserved.
 *
 */

#include "String.h"

String::String() {
	buf[0] = 0;
}

void String::write(uint8_t c) {
	char *p = buf;
	while (*p)
		p++;
	*p = c;
	p++;
	*p = 0;
}