// キャラクタ液晶のライブラリ
// (C)Copyright 2011 特殊電子回路

class LiquidCrystal {
private:
	char pinnum[11];
	void init();
	unsigned char cols;
	unsigned char rows;

	unsigned char col;
	unsigned char row;
	unsigned char entry_mode;
	unsigned char display_mode;
	unsigned char cursol_mode;
	unsigned char mode4bit;

	void send_control(unsigned char val);
	void send_data(unsigned char data);

public:
	int writeDelay; // １文字書き込み後の遅延時間(デフォルト0)

	LiquidCrystal(int rs, int enable, int d4, int d5, int d6, int d7);
	LiquidCrystal(int rs, int rw, int enable, int d4, int d5, int d6, int d7);
	LiquidCrystal(int rs, int enable, int d0, int d1, int d2, int d3, int d4, int d5, int d6, int d7);
	LiquidCrystal(int rs, int rw, int enable, int d0, int d1, int d2, int d3, int d4, int d5, int d6, int d7);
	void begin(int cols,int rows);
	void clear();
	void home() ;
	void setCursor(unsigned char col,unsigned char row) ;
	void write(unsigned char data);
	void print(const char *str);
	void print(int val);
	void cursor();
	void noCursor();
	void blink();
	void noBlink();
	void display();
	void noDisplay();
	void scrollDisplayLeft();
	void scrollDisplayRight();
	void autoscroll();
	void noAutoscroll();
	void leftToRight();
	void rightToLeft();
	void createChar(unsigned char location,const unsigned char charmap[]);
};
