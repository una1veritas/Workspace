/*
 *  String.h
 *  Demo
 *
 *  Created by ‰º‰’ ^ˆê on 10/05/23.
 *  Copyright 2010 ‹ãBH‹Æ‘åŠwî•ñHŠw•”. All rights reserved.
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