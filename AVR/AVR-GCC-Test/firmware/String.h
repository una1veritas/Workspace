/*
 *  String.h
 *  Demo
 *
 *  Created by ���� �^�� on 10/05/23.
 *  Copyright 2010 ��B�H�Ƒ�w���H�w��. All rights reserved.
 *
 */

#include "Print.h"

class String : public Print {
	char buf[256];
	
public:
	
	String();
	void write(uint8_t);
	using Print::write;
};

extern String sbuf;