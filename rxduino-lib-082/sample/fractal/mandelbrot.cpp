#include <rxduino.h>
#include <stdlib.h>

// OLEDƒ‰ƒCƒuƒ‰ƒŠ‚ðŽg‚¤
#include <oled.h>

int count = 0;
float mag;
float offsetx;
float offsety;

#define MARY1 1
#define MARY2 2

void setup()
{
  Serial.begin(38400);

  pinMode(PIN_SW,INPUT);
  pinMode(PIN_LED3,OUTPUT);

  Init_OLED(MARY2);
  OLED_printf_Font(OLED_FONT_MEDIUM);
  OLED_printf_Position(0, 0);
  OLED_printf_Color(OLED_WHT, OLED_BLK);

  OLED_printf_Font(OLED_FONT_MEDIUM);
  OLED_printf_Position(0, 1);
  OLED_printf_Color(OLED_WHT, OLED_BLK);
  OLED_printf(MARY2,"TOKUSHU");
  OLED_printf_Position(1, 3);
  OLED_printf_Color(OLED_YEL, OLED_BLK);
  OLED_printf(MARY2,"DENSHI");
  OLED_printf_Position(1, 5);
  OLED_printf_Color(OLED_BLU, OLED_BLK);
  OLED_printf(MARY2,"KAIRO");
  OLED_printf_Position(2, 7);
  OLED_printf_Color(OLED_CYN, OLED_BLK);
  OLED_printf(MARY2,"Inc.");

  delay(1000);
  offsetx = -0.756423894274328;
  offsety = 0.064179410646170;
  mag = 1;
}

void complex_mult(float ar,float ai,float br,float bi,float *cr,float *ci) {
  // C = A * B
  *cr = ar * br - ai * bi;
  *ci = ar * bi + ai * br;
}

float complex_abs2(float r,float i) {
  float ar,ai;
  complex_mult(r,i,r,-i,&ar,&ai);
  return ar;
}

int generateColor(int count, int base) {
  int r,g,b;
  int d = (count % base) * 256 / base;
  int m = (int)(d / 42.667);

  switch(m) {
    case 0: r=0;          g=6*d;          b=255;         break;
    case 1: r=0;          g=255;          b=255-6*(d-43);break;
    case 2: r=6*(d-86);   g=255;          b=0;           break;
    case 3: r=255;        g=255-6*(d-129);b=0;           break;
    case 4: r=255;        g=0;            b=6*(d-171);   break;
    case 5: r=255-6*(d-214); g=0;         b=255;         break;
    default: r=0;         g=0;            b=0;           break;
  }

  return (((r >> 3) & 0x1f) << 11) | (((g >> 2) & 0x3f) << 5) | ((b >> 3) & 0x1f);
}

void loop()
{
  OLED_move_position(MARY2,0,0);
  for(int y = 0;y < 128;y++) {
    for(int x = 0;x < 128;x++) {
      float cr = (x - 64) / 64. / mag + offsetx;
      float ci = (y - 64) / 64. / mag + offsety;
      float zr = 0;
      float zi = 0;
      int t;
      for(t=0;t<512;t++) {
        if(complex_abs2(zr,zi) > 4) break;
        complex_mult(zr,zi,zr,zi,&zr,&zi);
        zr += cr;
        zi += ci;
      }
      OLED_Send_Pixel(MARY2,generateColor(t,64));

    }
  }
  mag = mag * 1.2;
//  offsety += 0.001;
//  offsetx += 0.0016;

  if(count++ > 100)
  {
    count = 0;
    offsetx = -0.756423894274328;
    offsety = 0.064179410646170;
    mag = 1;
    count = 0;
  }
  for(int i=0;i<100;i++)
  {
    digitalWrite(PIN_LED3,i & 1);
    delayMicroseconds(300);
  }
}
