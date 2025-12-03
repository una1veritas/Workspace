/*
	GAME Language interpreter ,32bit Takeoka ver.
	by Shozo TAKEOKA (http://www.takeoka.org/~take/ )

	Modified by Akihito Honda at February 2023.

	- This source code was inpremented for SBCV20/8088.
	- This source code can compile with TURBO C Ver 2.01

*/

typedef unsigned char   u_char;
typedef unsigned short  u_short;
typedef unsigned int    u_int;
typedef unsigned long   u_long;

extern void memmove(void*, const void *, int);
extern void newline();
extern void strcpy();

extern char c_kbhit();
extern char c_getch();
extern void c_putch();
extern void srand();
extern short rand();
extern void warm_boot();
extern mach_fin();

#define CR 0x0D
#define LF 0x0A
#define BS 0x08
#define HT 0x09
#define DEL 0x7F
#define NULL 0x00
#define RAMEND 0x7fff

#define	MAX_STK	100
#define SIZE_LINE 160 /* Command line buffer length + NULL */
#define iSnum(c) ('0'<=(c) && (c)<='9')
#define iShex(c) (('0'<=(c) && (c)<='9')||('A'<=(c) && (c)<='F')||('a'<=(c) && (c)<='f'))

#define VARA(v) var[(v)]
#define	TOPP	VARA('=')	/* Top of program point */
#define	BTMP	VARA('&')	/* Bottom point (End of Program Area) */
#define	MOD	VARA('%')
#define REND VARA('*')

#define TEXT_SIZE 0x5d00	/* The value is desired after making binary */
void c_puts(const char *s) { while(*s) c_putch(*s++); }

/*****/
/*u_char text_buf[16384];*/
u_char text_buf[TEXT_SIZE];
u_char lin[SIZE_LINE];
u_char lky_buf[SIZE_LINE];
u_char *pc;
int  sp,stack[MAX_STK];
int lno;

/*	Var	*/
int var[128];

u_char *open_msg = "GAME86\r\nSBC-V20/8088 Edition\r\n";
u_char *rdy_msg = "*READY\r\n";
u_char *t_lock = "1";

main(u_char st_flg)
{
 int n,x, cnt;

	if ( !st_flg ) {
		srand(5678); /* for RND function */
		TOPP =(int)text_buf;
		REND = RAMEND;
		newText1();
		c_puts(open_msg);
	}
	c_puts(rdy_msg);

	for(;;){
		sp= -1;
		lno=0;
		cnt = c_gets(lin);	/* get a line */

		*((char *)(lin+cnt+1)) = 0x80; /* EOF on endOfLinebuf*/
		pc=lin;
		skipBlank();
		n=getNum(&x);
		if(x==0) {
			exqt();
			newline();
			c_puts(rdy_msg);
		}
		else edit(n);
	}
}

u_char* skipLine(p)
u_char *p;
{
	for(;*p;)
		p++;
	return p+1;
}

u_char* searchLine(n, f)
int n;
int *f;
{
	u_char* p;
	int l;

	for(p=(u_char*)TOPP;!(*p & 0x80);){
		l= (*p << 8) | *(p+1);
		if(n==l){ *f=1; return p;}
		if(n< l){ *f=0; return p;}
		p=skipLine(p+2);
	}
	*f=0;
	return p;
}

/* line edit routines */
edit(n)
int n;
{
	u_char *p;
	int f;

	if(n==0){ 
		dispList(TOPP);
		w_boot(NULL);
	}

	p=searchLine(n, &f);

	if(*pc=='/') {
		dispList(p);	/* list */
		w_boot(NULL);
	}
	else { 	/*edit */
		if(*((u_char*)BTMP) != 0xFF){
			er_boot(t_lock);
		}
		if(f) deleteLine(p);
		if(*pc=='\0') return 0; /* delete line */
		addLine(n,p,pc);
	}
	return 0;
}

addLine(n,p,new)
u_char *p,*new;
{
	int l;

	l= 2+ strlen(new)+1;
	memmove(p+l,p,(((u_char*)BTMP)-p)+1);
	*p= n>>8;
	*(p+1)= n;
	strcpy(p+2,new);
	BTMP += l;
}

deleteLine(p)
u_char *p;
{
	int l;

	l= 2+ strlen(p+2)+1;
	memmove(p,p+l,(((u_char*)BTMP)-p)-l+1);
	BTMP -= l;
}

int g_decStr(char *buf, unsigned int num) {
	int	cnt;
	char *b;
	
	cnt = 0;
	do {
		*buf++ = (char)((num % 10) | 0x30);
 		num = num / 10;
		cnt++;
	} while(num>0);
	*buf = '\0';
	return cnt;
}

void mk_dStr(char *d_buf, unsigned int num, int digit) {
	char s_buf[8];
	int	i, j, cnt, sign;

	sign = 0;
	if (num & 0x8000) {
		sign = 1;
		num = ~num + 1;		/* 2's complement */
	}

	cnt = g_decStr(s_buf, num);	/* get decimal 'upside down' string */
	j = cnt;
	if (sign) cnt++;

	i = 0;
	while (digit > cnt) {
		d_buf[i++] = ' ';
		digit--;
	}
	if (sign) d_buf[i++] = '-';
	
	while(j){
		d_buf[i++] = s_buf[j-1];
		j--;
	}
	d_buf[i] = '\0';
}

void g_hexStr(char *buf, u_short num, int cnt) {
	int i;
	char n;
	u_short msk;
	
	/* cnt = 2 or 4 */
	if( cnt == 4 ) msk = 0xf000;
	else msk = 0xf0;

	i= (cnt-1)*4;

	do {
		n = ((num & msk) >> i);
		msk = msk >> 4;
		if (n > 9) n += 55;
		else n +=48;
		*buf++ = (char)n;
		i -= 4;
	} while( i>=0 );
	*buf = '\0';
}

u_char *
dispLine(p)
u_char *p;
{
	int l;
	char s[8];

	l= (*p << 8) | *(p+1); p+=2;
	mk_dStr(s, l, 5);
	c_puts(s);
	for(;*p;)c_putch(*p++);
	newline();
	return p+1;
}

dispList(p)
u_char *p;
{
	for(;!(*p & 0x80);){
		breakCheck();
		p = dispLine(p);
	}
}

skipBlank()
{
	for(;;){
		if( *pc != ' ') return;
		pc++;
	}
}

skipAlpha()
{int x;
	for(;;){
		x= *pc;
		if((x<'A')||('z'<x)||('Z'<x && x<'a')) return x;
		pc++;
	}
}

/** execute statement **/

exqt()
{
	/* int c; */
	for(;;){
		skipBlank();
		do_cmd();
	}
}

topOfLine()
{
	int x,c;

more:
	x= *pc++;
	if(x & 0x80) w_boot(NULL);		/* force warm boot */

	lno = (x <<8)| *pc++;

	if(*pc != ' '){ /* Comment */
		pc=skipLine(pc);
		goto more;
	}
}

u_char *bk_msg = "\r\nStop!";

breakCheck()
{
	int c;

	if (c_kbhit()) {/* check keyin */
		c = c_getch();
		if(c == 0x03) w_boot(bk_msg);		/* force warm boot */
		if(c == 0x13) c_getch(); 			/*pause*/
	}
}

do_cmd()
{
	int c,c1,c2,e,vmode,off;

	breakCheck();
	c= *pc++;
	c1= *pc;
	switch(c){
		case '\0':
			topOfLine();
			return 1;
		case ']' :
			pc=(u_char*)pop();
			return 0;
		case '"' :
			do_pr();
			return 0;
		case '/' :
			newline();
			return 0;
		case '@' :
			if(c1=='=') {
				c2= *(pc+1);
				e=operand();
				do_until(e,c2);
				return 0;
			}
			do_do();
			return 0;
		case '?' :
			do_prNum(c1);
			return 0;
		case '\\' :
			mach_fin();
	}

	if(c1=='='){
		switch(c){
			case '#' :
				e=operand();do_goto(e);return 0;
			case '!' :
				e=operand();do_gosub(e);return 0;
			case '$' :
				e=operand(); c_putch(e);return 0;
			case '.' :
				e=operand();do_prSpc(e);return 0;
			case ';' :
				e=operand();do_if(e);return 0;
			case '\'' : /*RAND seed */
				e=operand();srand(e);return 0;
/*
			case '@' :
				c2= *(pc+1);e=operand();do_until(e,c2); return 0;
*/
			case '&' :
				e=operand();
				if(e==0)newText();
				return 0;
		}
	}
	vmode=skipAlpha();
	if(vmode==':' || vmode=='(' ){
		pc++;
		off=expr(*pc++);
		if(*(pc-1) !=')') er_boot("2");		/* force warm boot */
		e=operand();
		if ( vmode == ':') *(((u_char*)VARA(c)+off))=e; 
		else if ( vmode == '(') *(((u_short*)VARA(c)+off))=e;
		return 0;
	}
	e=operand();
	VARA(c)=e;

	if(*(pc-1)== ','){ /* For */
			c= *pc++;
			e= expr(c);
			push(pc);
			push(e);
	}
	return 0;
}

do_until(e,val)
{
	VARA(val)=e;
	if(e>stack[sp]){
		sp-=2; /*pop pc,value*/
		return ;
	}
	/* repeat */
	pc=(u_char*)stack[sp-1]; /*pc*/
	return ;
}

do_do()
{
	push(pc);
	push(0);
}

do_if(e)
{
	if(e==0){
		pc=skipLine(pc);
		topOfLine();
	}
}

do_goto(n)
{
	int f;
	u_char *p;

	if(n==-1) w_boot(NULL);		/* force warm boot */

	p=searchLine(n, &f);
	pc=p;
	topOfLine();
}

do_gosub(n)
{
	int f;
	u_char *p;

	p=searchLine(n, &f);
	push(pc);
	pc=p;
	topOfLine();
}

do_prSpc(e)
{
	int i;
	for(i=0;i<e;i++) c_putch(' ');
}

do_prNum(c1)
{
	int e,digit;

	if(c1== '('){
		pc++;
		digit=term(c1);
		e=operand();

		mk_dStr(lky_buf,e,digit);
		c_puts(lky_buf);
		return ;
	}

	e=operand();
	switch(c1){
	 	case '?' :
			g_hexStr(lky_buf, e, 4);
			break;
		case '$' :
			g_hexStr(lky_buf, e, 2);
			break;
		case '=' :
			mk_dStr(lky_buf, e, 1);
			break;
		default:
			er_boot("3");		/* force warm boot */
	}
	c_puts(lky_buf);
}

do_pr()
{
	int x;

	for(;;){
		if('"' == (x= *pc++)) break;
		if(x== '\0'){ pc--;break;}
		c_putch(x);
	}
}

pop()
{
	if(sp<0) er_boot("4");		/* force warm boot */
	return stack[sp--];
}

push(x)
{
	if(sp>=(MAX_STK-1)) er_boot("5");		/* force warm boot */
	return stack[++sp]=x;
}

operand()
{
	int x,e;

	for(;;){
		x= *pc++;
		if(x == '=') break;
		if(!(x & 0xDF)) errMsg(" ?");
	}
	x= *pc++;
	e= expr(x);
	return e;
}

char mm[4];

expr(c)
{
	int o,o1,op2;
	int e;
 
	e=term(c);

	for(;;){
		o= *pc++;
		switch(o){
			case '\0' :
				pc--;
			case ' ' :
			case ')' :
			case ',' :
				return e;
			case '<' :
				o1= *pc++;
				switch(o1){
					case '>' :
						op2=term(*pc++);
						e= (e!=op2);
						continue;
					case '=' :
						op2=term(*pc++);
						e=(e<=op2);
						continue;
					default:
						op2=term(o1);
						e=(e<op2);
						continue;
				}
			case '>' :
				o1= *pc++;
				switch(o1){
					case '=' :
						op2=term(*pc++);
						e=(e>=op2);
						continue;
					default:
						op2=term(o1);
						e=(e>op2);
						continue;
				}
			case '+' : op2=term(*pc++);e= e+op2;break;
			case '-' : op2=term(*pc++);e= e-op2;break;
			case '*' : op2=term(*pc++);e= e*op2;break;
			case '/' : op2=term(*pc++); MOD=e%op2; e= e/op2;break;
			case '=' : op2=term(*pc++);e= (e==op2);break;
			default:
				mm[0]=' ';mm[1]=o; mm[2]='?'; mm[3]=0;
				errMsg(mm);
		}
	}
}

term(c)
{
	int e,f=0, vmode;
	u_char *ppp;

	switch(c){
		case '$' :
			e= getHex(&f);
			if(f==0) return c_getch();
			return e;

		case '(' : /*EXPR */
			e=expr(*pc++);
			if(*(pc-1) !=')')errMsg(" )?");
			return e;

		case '+' : /*ABS */
			e= term(*pc++);
			return e<0? -e : e;

		case '-' : /* MINUS */
			return -(term(*pc++));

		case '#' : /* NOT */
			return !(term(*pc++));

		case '\'' : /*RAND */
			return (rand()%term(*pc++))+1;

		case '%' : /* MOD not yet*/
			e=term(*pc++);
			return MOD;

		case '?' : /*input */
			c_gets(lky_buf);	/* xgets(b); */
			ppp=pc;
			pc=lky_buf;
			e=expr(*pc++);
			pc=ppp;
			return e;

		case '"' : /*Char const */
			e = *pc++;
			if(*pc++ != '"') errMsg(" \"?");
			return e;
	}
	if(iSnum(c)){
		pc--; e= getNum(&f);
		return e;
	}

	vmode= skipAlpha();
	if(vmode==':' || vmode=='(' /*|| vmode=='['*/ ){
		pc++;
		e=expr(*pc++);
		
		if(*(pc-1) !=')') errMsg(" )?");	/* no return */

		switch(vmode){
			case ':' : return *(((u_char*)VARA(c)+e));
			case '(' : return *(((u_short*)VARA(c)+e));
		}
	}
	return VARA(c);
}

errMsg(s)
u_char *s;
{
	char a[8];

	c_puts("\r\nErr");
	c_puts(s);
	if(lno !=0) {
		c_puts(" in ");
		/*sprintf(b,"%d", lno);*/
		mk_dStr(a, lno, 1);
		c_puts(a);
 	}
	w_boot(NULL);		/* force warm boot */
}

int w_boot(unsigned char *msg)
{
	if (msg) c_puts(msg);
	warm_boot();
}

int er_boot(unsigned char *msg)
{
	c_puts("\r\nErr");
	if (msg) c_puts(msg);
	warm_boot();
}

char c_isprint(char c) {return(c >= 32  && c <= 126);}
char c_isspace(char c) {return(c == ' ' || (c <= 13 && c >= 9));}

void newline(void){
	const char *crlf = "\r\n";
	
	c_puts( crlf);
}

int c_gets(char *lbuf){
	char c;
	u_short len;
	
	len = 0;
	while((c = c_getch()) != CR){
		if( c == HT) c = ' '; /* TAB exchange Space */
		if(((c == BS) || (c == DEL)) && (len > 0)){ /* Backspace manipulation */
			len--;
			c_putch(BS); c_putch(' '); c_putch(BS);
		} else
		if(c_isprint(c) && (len < (SIZE_LINE - 1))){
			lbuf[len++] = c;
			c_putch(c);
		}
	}
	newline();
	lbuf[len] = 0;	/* Put NULL */

	if(len > 0) {
		do {
			--len;
		} while(c_isspace(lbuf[len]));
		++len;
		lbuf[len] = 0;	/* Put NULL */
	}
	return len;
}

void memmove(void *dest, const void *src, int n)
{
	char *tmp;
	const char *s;
 
	if (dest <= src) {
		tmp = (char *) dest;
		s = (char *) src;

		while (n--) {
			*tmp++ = *s++;
		}
	} else {
		tmp = (char *) dest;
		tmp += n;
		s = (char *) src;
		s += n;

		while (n--) {
			*--tmp = *--s;
		}
	}
}
 
void strcpy(char *p1, char *p2)
{

	/* lower to upper, except while string */
	register char c, flg;
	
	flg = 0;
	while( (c = *p2++) != '\0' ){
		if (c==0x22) flg ^= 1;
		if (c>=0x61 && c<=0x7a && !flg) c &= 0xdf;
		*p1++ = c;
	}
	*p1 = '\0';
}

int strlen(const char *s)
{
	int num = 0;
 
	while (*s++) {
		num++;
	}

	return num;
}

int getNum(f)
int *f;
{
	u_char c;
	int n=0;
	*f=0;

	c = *pc;
	while( iSnum(c) ) {
		n= n*10 + (int)(c-'0');
		pc++;
		c = *pc;
		*f=1;
	}
	return n;
	
}

int getHex(f)
int *f;
{
	int c;
	int n=0;
	*f=0;

	c= *pc;
	while(iShex(c)) {
		n= n*16 + (int)((c<'A')?(c-'0'):((c<'a')?(c-'A') :(c-'a'))+10 );
		pc++;
		c= *pc;
		*f=1;
	}
	return n;
}

newText()
{
	if( *((u_char*)BTMP) != 0xFF) er_boot(t_lock);		/* force warm boot */
	newText1();
}

newText1()
{
	BTMP = TOPP;
	*((u_char*)BTMP) = 0xFF;
}
