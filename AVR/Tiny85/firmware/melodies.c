/*
 *  Melodies.h
 *  Tiny85
 *
 *  Created by â∫âí ê^àÍ on 10/07/11.
 *  Copyright 2010 ã„èBçHã∆ëÂäwèÓïÒçHäwïî. All rights reserved.
 *
 */

#include <avr/pgmspace.h>

#include "wiring.h"
#include "melodies.h"


byte testsong[][2] PROGMEM= {
   {25, 30},
   {25, 8},
   {0, 7},
   {25, 8},
   {0, 7},
   {25, 90},
   {0, 30},
   {25, 20},
   {27, 20},
   {20, 20},
   {25, 60},
   {30, 20},
   {29, 20},
   {30, 20},
   {32, 60},
   
   {32, 40},
   {32, 20},
   {32, 20},
   {34, 20},
   {30, 20},
   {32, 120},
   
   {29, 30},
   {37, 30},
   {30, 30},
   {39, 30},
   {32, 30},
   {32, 8},
   {0, 7},
   {32, 8},
   {0, 7},
   {37, 180},
   {0,  60},
   {0, 0}
};

byte mmmarch[][2] PROGMEM = {
   {18,30},
   {0, 30},
   {13,30},
   {0, 30},
   {15,30},
   {0, 30},
   {17,30},
   {0, 30},
   {18,30},
   {0, 30},
   {13,30},
   {0, 30},
   {15,30},
   {0, 30},
   {17,30},
   {0, 30},
   
   {18, 35},
   {30,  10},
   {18, 15},
   {18, 35},
   {29,  10},
   {18, 15},
   {18, 35},
   {27,  10},
   {18, 15},
   {18, 35},
   {25,  10},
   {18, 15},
   {20, 35}, 
   {23, 10},
   {18, 15}, 
   {17, 35}, 
   {20, 10},
   {15, 15}, 
   {25, 10},
   {13, 50}, 
   
   {0, 60},
   
   {18, 45},
   {18, 15},
   {18, 45},
   {18, 15},
   {18, 45},
   {18, 15},
   {18, 45},
   {0,	 15},
   {22, 45}, 
   {18, 15}, 
   {20, 45}, 
   {17, 15}, 
   {18, 45},
   {0,  15},
   
   {18, 45},
   {17, 15},
   {15, 60},
   {27, 20},
   {30, 20},
   {27, 20},
   {39, 60},
   {18, 45},
   {16, 15},
   {13, 60},
   {25, 20},
   {30, 20},
   {25, 20},
   {37, 60},
   {0, 60},
   
   {15, 45},
   {15, 15},
   {15, 45},
   {15, 15},
   {15, 45},
   {15, 15},
   {17, 45},
   {18, 15},
   {20, 60},
   {13, 30},
   {0,  30},
   
   {25, 20},
   {27, 20},
   {25, 20},
   {23, 20},
   {22, 20},
   {20, 20},
   
   {13, 3},
   {15, 3},
   {17, 4},
   {18, 25},
   {30,  10},
   {18, 15},
   {18, 35},
   {29,  10},
   {18, 15},
   {18, 35},
   {27,  10},
   {18, 15},
   {18, 35},
   {25,  10},
   {18, 15},
   {20, 35}, 
   {23, 10},
   {18, 15}, 
   {17, 35}, 
   {20, 10},
   {15, 15}, 
   {25, 10},
   {13, 50}, 
   
   {0, 60},
   
   {18, 45},
   {18, 15},
   {18, 45},
   {18, 15},
   {18, 45},
   {18, 15},
   {18, 45},
   {0,	 15},
   {22, 45}, 
   {18, 15}, 
   {20, 45}, 
   {17, 15}, 
   {18, 45},
   {0,  15},
   
   {0,0}
};


byte bqhd[][2] PROGMEM = {

   {25, 30},
   {29, 15},
   {30, 15},
   {32, 30},
   {25, 30},
   {22, 30},
   {27, 15},
   {25, 15},
   {24, 15},
   {22, 15},
   {24, 15},
   {20, 15},
   
   {25, 30},
   {25, 15},
   {27, 15},
   {29, 15},
   {27, 15},
   {29, 15},
   {25, 15},
   {27, 15},
   {29, 15},
   {27, 15},
   {25, 15},
   {24, 15},
   {22, 15},
   {24, 15},
   {20, 15},
   
   {25, 30},
   {29, 15},
   {30, 15},
   {32, 30},
   {25, 30},
   {22, 30},
   {27, 15},
   {25, 15},
   {24, 15},
   {22, 15},
   {24, 15},
   {20, 15},
   
   {25, 15},
   {27, 15},
   {29, 15},
   {25, 15},
   {27, 15},
   {29, 15},
   {27, 15},
   {25, 15},
   {24, 15},
   {22, 15},
   {24, 15},
   {20, 15},
   {25, 30},
   {0,  30},
   
   {29, 60},
   {27, 30},
   {20, 30},
   {25, 30},
   {15, 30},
   {24, 60},
   {29, 15},
   {31, 15},
   {32, 30},
   {27, 15},
   {20, 15},
   {24, 15},
   {20, 15},
   {25, 15},
   {15, 15},
   {22, 15},
   {15, 15},
   {20, 60},
   
   {17, 60},
   {15, 30},
   {20, 30},
   {22, 30},
   {15, 30},
   {20, 60},
   {29, 15},
   {30, 15},
   {29, 15},
   {25, 15},
   {27, 15},
   {29, 15},
   {27, 15},
   {24, 15},
   {25, 15},
   {27, 15},
   {25, 15},
   {22, 15},
   {20, 60},
   
   {25, 30},
   {29, 15},
   {30, 15},
   {32, 30},
   {25, 30},
   {22, 30},
   {27, 15},
   {25, 15},
   {24, 15},
   {22, 15},
   {24, 15},
   {20, 15},
   
   {25, 30},
   {25, 15},
   {27, 15},
   {29, 15},
   {27, 15},
   {29, 15},
   {25, 15},
   {27, 15},
   {29, 15},
   {27, 15},
   {25, 15},
   {24, 15},
   {22, 15},
   {24, 15},
   {20, 15},
   
   {25, 30},
   {29, 15},
   {30, 15},
   {32, 30},
   {25, 30},
   {22, 30},
   {27, 15},
   {25, 15},
   {24, 15},
   {22, 15},
   {24, 15},
   {20, 15},
   
   {25, 15},
   {27, 15},
   {29, 15},
   {25, 15},
   {27, 15},
   {29, 15},
   {27, 15},
   {25, 15},
   {24, 15},
   {22, 15},
   {24, 15},
   {20, 15},
   {25, 30},
   {0,  30},
   
   {0, 0}
};

byte bibidi[][2] PROGMEM = {
   //	{0, 30},
   {20, 20},
   {19, 20},
   {20, 20},
   {22, 40},
   {20, 20},
   {17, 20},
   {16, 20},
   {17, 20},
   {18, 40},
   {17, 20},
   
   {13, 20},
   {13, 20},
   {13, 20},
   {13, 20},
   {12, 20},
   {10, 20},
   {8,  60},
   {0,  60},
   
   {8,  20},
   {10, 20},
   {12, 20},
   {13, 20},
   {15, 20},
   {17, 20},
   {18, 20},
   {17, 20},
   {15, 20},
   {20, 60},
   
   {20, 20},
   {22, 20},
   {20, 20},
   {18, 20},
   {17, 20},
   {15, 20},
   {13, 25},
   {17, 7},
   {20, 7},
   {25, 7},
   {29, 7},
   {32, 7},
   {37, 30},
   {0, 30},
   
   {30, 20},
   {30, 20},
   {30, 20},
   {27, 45},
   {25, 15},
   {30, 60},
   {22, 45},
   {21, 15},

   {29, 20},
   {29, 20},
   {29, 20},
   {27, 45},
   {25, 15},
   {29, 30},
   {0, 30},
   {17, 45},
   {16, 15},
   
   {15, 20},
   {15, 20},
   {15, 20},
   {15, 45},
   {15, 15},
   {15, 45},
   {15, 15},
   {15, 45},
   {13, 15},
   
   {12, 20},
   {11, 20},
   {12, 20},
   {13, 20},
   {12, 20},
   {13, 20},
   {15, 60},
   {0, 60},

   {20, 20},
   {19, 20},
   {20, 20},
   {22, 40},
   {20, 20},
   {17, 20},
   {16, 20},
   {17, 20},
   {18, 40},
   {17, 20},
   
   {13, 20},
   {13, 20},
   {13, 20},
   {13, 20},
   {12, 20},
   {10, 20},
   {8,  60},
   {0,  60},
   
   {8,  20},
   {10, 20},
   {12, 20},
   {13, 20},
   {15, 20},
   {17, 20},
   {18, 20},
   {17, 20},
   {15, 20},
   {20, 60},
   
   {20, 20},
   {22, 20},
   {20, 20},
   {18, 20},
   {17, 20},
   {15, 20},
   {13, 25},
   {17, 7},
   {20, 7},
   {25, 7},
   {29, 7},
   {32, 7},
   {37, 30},
   {0, 30},
   
   {0, 0}
};
byte * melody(byte m) {
   switch(m) {
	case 1:
	   return (byte *) mmmarch;
	case 2:
	   return (byte *) bqhd;
	case 3:
	   return (byte *) bibidi;
	default:
	   return (byte *) testsong;
   }
}