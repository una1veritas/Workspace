/*
 *  String.cpp
 *  Demo
 *
 *  Created by ���� �^�� on 10/05/23.
 *  Copyright 2010 ��B�H�Ƒ�w���H�w��. All rights reserved.
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