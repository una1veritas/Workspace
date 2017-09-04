typedef struct {
  long x;
  long y;
  
  void init() {
    x = 0;
    y = 0;
  }

  void print(){
    printf("x=%d y=%d\n",x,y);
  }
} point;


class edge{
public:
  short line;
  long number;
  edge(){
    	line=0;
        number=0;
  }
  void set(long ln,long num){
    line=(short)ln;
    number=num;
  }
  void print(void){
    printf("%d-%d \n",
	   line,number);
    
  }
  friend int operator==(edge,edge);
    friend int operator!=(edge,edge);
};
int operator==(edge left,edge right){
        return ((left.line==right.line)&&(left.number==right.number));
}
int operator!=(edge left,edge right){
  return ((left.line!=right.line)||(left.number!=right.number));
}

class conection{
  
  edge start,end;
  
public:
  void init() {
    start.line=0;
    end.line=0;
    start.number=0;
    end.number=0;
  }
  
  long s_line(){
    return start.line;
  }
  long e_line(){
    return end.line;
  }
  long s_number(){
    return start.number;
  }
  long e_number(){
    return end.number;
  }
  void s_line(long x){
    start.line=(short)x;
  }
  void e_line(long x){
    	end.line=(short)x;
  }
  void s_number(long x){
    start.number=x;
  }
    void e_number(long x){
      end.number=x;
    }
    edge st(){
      return start;
    }
    edge en(){
      return end;
    }

    void sort(){

		if(start.line > end.line){

        	swap(&start.line,&end.line);
            swap(&start.number,&end.number);
        }else if((start.line == end.line) && (start.number > end.number)){
        	swap(&start.number,&end.number);
        }
    }
    void set(long l1,long n1,long l2, long n2){
		start.line=(short)l1;
        start.number=n1;
        end.line=(short)l2;
        end.number=n2;
        sort();
    }


    void print(void){
		printf("%d-%d %d-%d \n",
				start.line,start.number,end.line,end.number);

    }

    friend int operator==(conection,conection);
    friend int operator>(conection,conection);

};

    int operator>(conection left,conection right){
        if(left.start.line > right.start.line){
	        return 1;
        }else if(left.start.line == right.start.line){
         	if(left.start.number > right.start.number){
    		    return 1;
			}else if(left.start.number == right.start.number){
	          	if(left.end.line > right.end.line){
        			return 1;
                }else if(left.end.line == right.end.line){
                   	if(left.end.line > right.end.line){
        				return 1;
                    }else{
				        return 0;
					}
            	}
			}
        }
        return 0;
    }
    int operator==(conection left,conection right){
        return (((left.start.line==right.start.line)&&(left.end.line==right.end.line)&&
    	(left.start.number==right.start.number)&&(left.end.number==right.end.number))||
        ((left.start.line==right.end.line)&&(left.end.line==right.start.line)&&
        (left.start.number==right.end.number)&&(left.end.number==right.start.number)));

    }
// 2バイトであらわすとき．
class conection2{

	unsigned char start,end;

public:
    conection2(){
    	init();
    }
	void change_conection(conection x){
    	start=(unsigned char)((x.s_line()-1)*pow(2,6)+x.s_number());
    	end=(unsigned char)((x.e_line()-1)*pow(2,6)+x.e_number());
	   //	printf("start=%x end=%x s_line=%d e_line=%d s_num=%d e_num=%d\n",
	   //			start,end,(x.s_line()-1)*(long)pow(2,6),(x.e_line()-1)*(long)pow(2,6),x.s_number()-1,x.e_number()-1);
    }

    void init(){
    	start=0;
        end=0;
    }

    void set(unsigned char startt,unsigned char endd){
    	start=startt;
        end=endd;
    }

    unsigned char st(){
    	return start;
    }

    unsigned char en(){
    	return end;
    }

    long s_line(){
        if(start==0){
        	return 0;
        }else{
	  return start / (1<<6) /* pow(2,6)*/ +1 ;
        }
    }
    long e_line(){
        if(end==0){
        	return 0;
        }else{
	  return end / (1<<6) /* pow(2,6) */ +1;
        }
    }
    long s_number(){
    	return start & 0x3f;//; -(s_line()-1)*pow(2,6)  ;
    }
    long e_number(){
    	return end & 0x3f;//-(e_line()-1)*pow(2,6) ;
    }

    void set(long l1,long n1,long l2,long n2){
    	conection tmp;
        tmp.set(l1,n1,l2,n2);
    	change_conection(tmp);
    }

    long value(){
       return start*(long)pow(2,8)+end;
    }
    void print(void){
		printf("start=%x end=%x \n",
				start,end);
    }
    void print2(void){
		printf("%d-%d %d-%d \n",
				s_line(),s_number(),e_line(),e_number());

    }

    friend int operator==(conection2,conection2);
    friend int operator!=(conection2,conection2);
    friend int operator>(conection2,conection2);
    friend int operator<(conection2,conection2);
};

    int operator==(conection2 left,conection2 right){
        return ((left.st()==right.st())&&(left.en()==right.en()));

    }
    int operator!=(conection2 left,conection2 right){
        return ((left.st()!=right.st())||(left.en()!=right.en()));

    }

    int operator>(conection2 left,conection2 right){
        return left.value()>right.value();

    }

    int operator<(conection2 left,conection2 right){
        return left.value()<right.value();

    }


class dissect_conection:public conection2{
	public:
    int place;
    void init2(){
    	init();
        place=0;
    }
    void print3(void){
		printf("%d-%d %d-%d \n",
				s_line(),s_number(),e_line(),e_number());
    }
    friend int operator==(dissect_conection,conection2);
};
    int operator==(dissect_conection left,conection2 right){
        return ((left.st()==right.st())&&(left.en()==right.en()));

    }

int sort_function(conection2 *a,conection2 *b){
	if(*a>*b){
       	return 1;
    }else if(*a==*b){
       	return 0;
    }else{
       	return -1;
    }
}

class setOfconection{
private:
  conection2 *con;
public:
  setOfconection(){

    con = new conection2 [2*r];

  }
  
  ~setOfconection() {
      delete[] con;
  }
  
  void init(){
    for(long i=0;i<=2*r-1;i++){
      (con+i)->init();
    }
  }
  void set(long i,conection2 x){
    con[2*r-i]=x;
  } 
  void set2(long i,conection2 x){
    con[i]=x;
  }
  conection2 co(long x){
    return con[x] ;
  }
  conection2 co2(long x){
    return con[2*r-x] ;
  }
  
  void sort(){
    qsort((conection2 *)con, 2*r, sizeof(conection2), sort_function);
  }
  
long element_num(){
  long ans=0;
  conection2 dummy;
  dummy.init();
  
  for(long i=0;i<=2*r-1;i++){
    if(con[i]!=dummy){
      ans++;
    }
  }
  return ans;
}
long value(){
  long ans=0,powpow,max_pow;
  max_pow=(sizeof(long)/sizeof(conection2))*8;
  for(long i=0;i<=2*r-1;i++){
    powpow=(2,i % max_pow);
    ans += con[i].value()*powpow;
  }
  return ans;
}


long hash(long base){
  return value() % base;
}

void print(){
  printf("factor=");
  for(long i=0;i<=2*r-1;i++){
    printf("%x ",con[i].value());
  }
  printf("\n");
}
    void print2(){
      printf("factor2======\n");
      for(long i=0;i<=2*r-1;i++){
	con[i].print();
      }
      printf("\n");
    }
    void print3(){
      printf("factor3======\n");
      for(long i=0;i<=2*r-1;i++){
	con[i].print2();
      }
      printf("\n");
    }

    friend  int operator==(setOfconection & ,setOfconection &);
    
} ;


 int operator==(setOfconection & left,setOfconection & right){
  for(long i=0;i<=2*r-1;i++){
    if(left.co(i)!=right.co(i)){
      return 0;
            }
  }
  return 1;
}


class point_conection {
    long current_point;
    long line_number;
    public:
    conection2 line;
    conection link;

    point_conection(){
        current_point=0;

    }
    ~point_conection(){

    }
    void point_set(long po){
    	current_point=po;
    }
    void set_line_num(long x){
      line_number=x;
    }
    int line_num(){
      return line_number;
    }
    long c_point(){
      return current_point;
    }
};

class answer_factor {
	public:
	double length;
    point_conection *p_cone;

    answer_factor(){
    	length=-1.0;
    }  
    ~answer_factor(){
    }
    void set(long len,point_conection *p){
    	length=len;
        p_cone=p;
    }
    void print(long point_num){
    	printf("length=%f",length);
        for(long i=0;i<=point_num-1;i++){
            printf("current_point=%d :",p_cone[i].c_point());
        	p_cone[i].line.print2();
			p_cone[i].link.print();
        }
        printf("\n");
	}


    friend int operator==(answer_factor left,answer_factor right);
    friend int operator!=(answer_factor left,answer_factor right);
    friend int operator>(answer_factor left,answer_factor right);
    friend int operator<(answer_factor left,answer_factor right);
};
    int operator==(answer_factor left,answer_factor right){
        return left.length==right.length;

    }
    int operator!=(answer_factor left,answer_factor right){
        return left.length!=right.length;

    }

    int operator>(answer_factor left,answer_factor right){
        return left.length>right.length;

    }

    int operator<(answer_factor left,answer_factor right){
        return left.length<right.length;

    }

class answer_factor2{
	public:
	double length;
    point_conection *p_cone;
    answer_factor2(){
      p_cone = new point_conection[Maxpoint];
    	length=-1.0;
    }
     ~answer_factor2(){
       delete[] p_cone;
    }
    void set(long len,point_conection *p){
      length=len;
      for(long i=0;i<=Maxpoint-1;i++){
      p_cone[i]=p[i];
      }
    }
    void print(long point_num){
    	printf("length=%f\n",length);
        for(long i=0;i<=point_num-1;i++){
            printf("line_number=%d current_point=%d :",p_cone[i].line_num(),p_cone[i].c_point());
        	p_cone[i].line.print2();
			p_cone[i].link.print();
        }
        printf("\n");
	}
    operator answer_factor(){
    	answer_factor ans;
        conection2 dummy;
        long i=0;
        ans.length=length;
        while(p_cone[i].line!=dummy || i<=Maxpoint){
        	ans.p_cone[i]=p_cone[i];
           	i++;
        }
        return ans;
    }
    answer_factor2(const answer_factor2&);
    answer_factor2& operator=(const answer_factor2&);
    friend const  int operator==(const answer_factor2 &left,const answer_factor2 &right);
    friend const int operator!=(const answer_factor2 &left,const answer_factor2 &right);
    friend const int operator>(const answer_factor2 &left,const answer_factor2 &right);
    friend const int operator<(const answer_factor2 &left,const answer_factor2 &right);
};

answer_factor2 :: answer_factor2(const answer_factor2 &a){
  length=a.length;
  p_cone = new point_conection[Maxpoint];
  for(long i=0;i<=Maxpoint-1;i++){
    p_cone[i]=a.p_cone[i];
  }
}
const int operator==(const answer_factor2 &left,const answer_factor2 &right){
  return left.length==right.length;
  
}
const int operator!=(const answer_factor2 &left,const answer_factor2 &right){
  return left.length!=right.length;
  
}

const int operator>(const answer_factor2 &left,const answer_factor2 &right){
  return left.length>right.length;

}

const int operator<(const answer_factor2 &left,const answer_factor2 &right){
  return left.length<right.length;
  
}

answer_factor2& answer_factor2:: operator=(const answer_factor2 &a){
  length=a.length;
  for(long i=0;i<=Maxpoint-1;i++){
    p_cone[i]=a.p_cone[i];
  }

  return *this;
}

class sub_root:public answer_factor{
	public:
	setOfconection cone;
	~sub_root(){

	}
	
    void print(){
    	printf("length=%f",length);
        cone.print();

	}
    void print2(long point_num){
    	printf("length=%f\n",length);
        cone.print();
        for(long i=0;i<=point_num-1;i++){  
	  printf("line_number=%d current_point=%d :",p_cone[i].line_num(),p_cone[i].c_point());
	  p_cone[i].line.print2();
	  p_cone[i].link.print();
        }
    }

};



// 各辺が何回分割されたかをもとめ値としてもつ
class line_level{
	public:
    long upper_level,lower_level,left_level,right_level;
    long min_level,max_level;

	// 座標xはレヴェルいくつの分割線か求める
    long search_level(long);
    //　各辺のレヴェルをもとめる
    void set_level(long,long ,long ,long );
    void print_lev(void);
};

long line_level::search_level(long x){
  long tmp=L;
    long ans=0;
    while(x % tmp){
       	tmp /= 2;
            ans++;
           }
       return ans;
}


    //　各辺のレヴェルをもとめる
void line_level::set_level(long x1,long y1,long x2,long y2){

   	upper_level=search_level(y1);
   	lower_level=search_level(y2);
   	left_level=search_level(x1);
   	right_level=search_level(x2);
    if (upper_level>=lower_level){
    	max_level=upper_level;
        min_level=lower_level; //erase
    }else{
    	max_level=lower_level;
        min_level=upper_level; //erase
    }
    //erase
    if(left_level<=min_level){min_level=left_level;}
    if(right_level<=min_level){min_level=right_level;}

}
void line_level::print_lev(void){
	printf("upper=%d lower=%d left=%d right=%d \n"
   	   ,upper_level,lower_level,left_level,right_level);
}


//　穴のあき方
//　ひとつの正方形に対してもっとも高いレベルの分割線と比べて
//　残りの二辺はどれだけレベルが離れているかまた，また正方形は
//　それぞれのレヴェルの分割線の中心から何番目なのかをもとめる．
class portals:public line_level {

    public:
  long level[5];
  long number[5];
  long portals_num[5];
  long total_line_type;


    long search_total_line_type(void);
    long search_total_line_num(long);
    long search_portals_num(long ,long );
    //　 levelの分割線の中心から何番目なのかをもとめる
    int search_number(long ,long ,long );
    //　それぞれの値を求める
    void serach(long ,long , long , long );
    //　求めた穴のあき方がどのタイプの正方形の穴のあき方か求める関数
    long  search_type(void);

    long total_line(long,long);

     void print();

};


long portals::search_total_line_num(long i){
		long edge1,edge2;

            return_edge(4,(long)i,&edge1,&edge2);
            if(edge1==edge2){
            	return sum(portals_num[edge1]-1,1,portals_num[edge1]-1)+portals_num[edge1];
            }else{
                return portals_num[edge1]*portals_num[edge2];
            }
	}

long portals::search_total_line_type(void){
  long ans=0;
  long i[11],l_num[11];
  for( i[1]=0;i[1]<=sum_H(total_line(1,1),r/2);i[1]++){
    l_num[1]=search_line_num(i[1],total_line(1,1));
    for( i[2]=0;i[2]<=sum_H2(total_line(1,2),r-l_num[1]*2,r);i[2]++){
      l_num[2]=search_line_num(i[2],total_line(1,2));
      for( i[3]=0;i[3]<=sum_H2(total_line(1,3),r-l_num[1]*2-l_num[2],r);i[3]++){
	l_num[3]=search_line_num(i[3],total_line(1,3));
	for( i[4]=0;i[4]<=sum_H2(total_line(1,4),r-l_num[1]*2-l_num[2]-l_num[3],r);i[4]++){
	  l_num[4]=search_line_num(i[4],total_line(1,4));
	  for( i[5]=0;i[5]<=sum_H(total_line(2,2),(r-l_num[2])/2);i[5]++){
	    l_num[5]=search_line_num(i[5],total_line(2,2));
	    for( i[6]=0;i[6]<=sum_H2(total_line(2,3),r-l_num[2]-l_num[5]*2,r-l_num[3]);i[6]++){
	      l_num[6]=search_line_num(i[6],total_line(2,3));
	      for( i[7]=0;i[7]<=sum_H2(total_line(2,4),r-l_num[2]-l_num[5]*2-l_num[6],r-l_num[4]);i[7]++){
		l_num[7]=search_line_num(i[7],total_line(2,4));
		for( i[8]=0;i[8]<=sum_H(total_line(3,3),(r-l_num[3]-l_num[6])/2);i[8]++){
		  l_num[8]=search_line_num(i[8],total_line(3,3));
		  for( i[9]=0;i[9]<=sum_H2(total_line(3,4),r-l_num[3]-l_num[6]-l_num[8]*2,r-l_num[4]-l_num[7]);i[9]++){
		    l_num[9]=search_line_num(i[9],total_line(3,4));
		    for(  i[10]=0;i[10]<=sum_H(total_line(4,4),(r-l_num[4]-l_num[7]-l_num[9])/2);i[10]++){
		      ans++;
		      
		    }
		  }
		}
	      }
	    }
	  }
	}
      }
    }
  }
  
  return ans;
}

long portals::search_portals_num(long level,long number){
  double l,tmp1,tmp2;
  l= 1.0/pow(2,level);
  tmp1 = l*(double)((number-1)*(m+1));
  tmp2 = l*(double)(number*(m+1));
  if(tmp1 <= 0){
    tmp1 = tmp1+1.0;
  }
  
  return long(ceil(tmp2-1.0) -ceil(tmp1) + 1);

}


    //　 levelの分割線の中心から何番目なのかをもとめる
int portals::search_number(long level,long a,long b){
    int i=0,ans;
	long l,diff,ab,center;
    l = (long)((double)L / pow(2.0,(double)level)*2.0) ;
    diff = (long)(labs((long)(a-b)));
    ab=(a+b)/2;
    while(l*(i+1) <= ab){
       	i++;
       }
    center= i*l+l/2;
    ans = (int)(ceil((double)labs((long int)(ab-center))/(double)diff)) ;
	return ans;
}

    //　それぞれの値を求める
void portals::serach(long x1,long y1, long x2, long y2){
	set_level(x1,y1,x2,y2);
  	level[1]=upper_level-lower_level;
   	level[2]=right_level-left_level;
  	level[3]=0;
   	level[4]=0;
    if(level[1] >=0){
      	number[1] = search_number(lower_level,x1,x2);
    }else{
       	number[1] = search_number(upper_level,x1,x2);
       	level[1] = abs(level[1]);
    }

    if(level[2] >=0){
       	number[2] = search_number(left_level,y1,y2);
    }else{
       	number[2] = search_number(right_level,y1,y2);
       	level[2] = abs(level[2]);
    }
    number[3]=1;
    number[4]=1;

    /*// 乱数を用いるとき消すところ
    if(y1==0 || y2==L){
    portals_num[1]= 0;
    }else{
    portals_num[1]=search_portals_num(level[1],number[1]);
    }
    if(x1==0 || x2==L){
    portals_num[2]=1;
    }else{
    portals_num[2]=search_portals_num(level[2],number[2]);
    }
    if(x1==0 && x2==L){
    portals_num[3]=0;
    }else{
    portals_num[3]=m;
    }

    if(y1==0 && y2==L){
    portals_num[4]=0;
    }else{
    portals_num[4]=m;
    }
     乱数を用いるとき戻すところ */
    portals_num[1]=search_portals_num(level[1],number[1]);
    portals_num[2]=search_portals_num(level[2],number[2]);
    portals_num[3]=m;
    portals_num[4]=m;
    //*/
    for(int i=1;i<=10;i++){
       //	total_line_num[i]=search_total_line_num(i);
        }
    //set_total_line_num();
    total_line_type=search_total_line_type();
    /*
    diff1= search_diff(level[1],number[1]);
    diff2= search_diff(level[2],number[2]);
      */
}

    //　求めた穴のあき方がどのタイプの正方形の穴のあき方か求める関数
long  portals::search_type(void){
   	long ans,tmp1,tmp2;
   	if(level[1] == 0){
       	return (0);
       }

    if(level[1] == level[2]){
       	tmp1 =level[1];
        if(number[1] == 1 && number[2] == 1){
   	        tmp2 = 1;
       	}else if(number[1] == number[2] ){
           	tmp2 = number[1] + 1;
        }else{
           	tmp2 =2;
        }
	}else if(level[1] > level[2]){
       	tmp1 =level[1];
        tmp2 = number[1] + 1;
       }else{
       	tmp1 =level[2];
        tmp2 =number[2] + 1;
	}

	ans = (long)(pow(2,tmp1) + tmp1 + tmp2 -3);
    //乱数を用いるとき消すところ．
    if(min_level==0 ){
    //ans++;
    }
    //ここまで
    return ans;
}

long portals::total_line(long i,long j){
   	return search_total_line_num(return_num(4,i,j));
   }
void portals::print(){

  printf("level[1]=%d level[2]=%d number[1]=%d number[2]=%d \n",
	 level[1],level[2],number[1],number[2]);
  printf("portals_num[1]=%d num[2]=%d num[3]=%d num[4]=%d \n",
	 portals_num[1],portals_num[2],portals_num[3],portals_num[4]);
}




//　分割された正方形のデータ
class quad_tree:public portals{

public:
  long x1,y1,x2,y2;   	  //　正方形の角の座標
  int place[5]; //　正方形が分割されてできる正方形のタイプ
  int point_num;                   //　正方形内の点の数
  long type;                        //　正方形のタイプ
  int type2;
  int dissection_line;             //　枠が正方形を通過しているかのフラグ

    double search_point(int,long);
    //　正方形の中に何個点があるかもとめる
    int search_point_num(point *);
    long search_point_num2(point *);
    point search_point_place(point *);
    int search_type2(void);
    //　枠が正方形を通過しているか調べる
    int search_dissection_line(point);
   //　各値をセットする
	void set(long ,long ,long ,long ,point*, point );
    void print_tree(void);
    long hash_table_size();
    };


double quad_tree::search_point(int line_num,long point_num){
  double ans;
  long l;
  long tmp1;
  double tmp3;
  tmp3=(double)(L/(double)(m+1));
  if(line_num<=2){
    l =  ((number[line_num]-1+(number[line_num]%2)) 
	  * ( L / (1L << level[line_num] ) ));
    
    tmp1= l / (long)tmp3;
     if(((double)(l/tmp3) ==ceil(l/tmp3))){
      tmp1--;
    }
    //  	printf("number=%d tmp3=%f l=%d tmp1=%d max_level=%d\n",number[line_num],tmp3,l,tmp1,max_level);
    if((number[line_num]%2)==1){
      ans=l-(tmp1-point_num+1)*tmp3;
    }else{
    //   	printf("(tmp1+point_num)*tmp3=%f l=%f\n",(double)(tmp1+point_num)*tmp3,(double)l);
      ans=(tmp1+point_num)*tmp3-l;
    }
    //   printf("level[line_num]=%d pow(2,level[line_num])=%f \n",level[line_num],pow(2,level[line_num]));
    return ans*pow(2,level[line_num]);
  }else{
    return (double)point_num*tmp3;
  }
}

//　正方形の中に何個点があるかもとめる
int quad_tree::search_point_num(point p[]){
  int i,ans=0;
  for(i=1;i<=Maxpoint;i++){
        	if ((x1 <=p[i].x && p[i].x < x2 )&& (y1 <=p[i].y && p[i].y < y2) ){
		  ans++;
            }
  }
  return ans ;
}

  long quad_tree::search_point_num2(point p[]){
    	int i;
       	for(i=1;i<=Maxpoint;i++){
        	if ((x1 <=p[i].x && p[i].x < x2 )&& (y1 <=p[i].y && p[i].y < y2) ){
                  	return i;
            }
        }
        return 0;
    }

  point quad_tree::search_point_place(point p[]){
    	int i;
        point ans;
       	for(i=1;i<=Maxpoint;i++){
        	if ((x1 <=p[i].x && p[i].x < x2 )&& (y1 <=p[i].y && p[i].y < y2) ){
            		if(type2==1){
                  	ans.x=x2-p[i].x;
                  	ans.y=y2-p[i].y;
                    }else if(type2==2){
                  	ans.x=x2-p[i].x;
                  	ans.y=p[i].y-y1;

                    }else if(type2==3){
                  	ans.x=p[i].x-x1;
                  	ans.y=y2-p[i].y;
                    }else if(type2==4){
                  	ans.x=p[i].x-x1;
                  	ans.y=p[i].y-y1;
                    }
                    //ans.x *= pow(2.0,(double) max_level);
                    //ans.y *= pow(2.0,(double) max_level);
                    ans.x *= (1 << max_level);
                    ans.y *= (1 << max_level);

            }
        }
        return ans ;
    }
int quad_tree::search_type2(void){
    if (upper_level>=lower_level){
       if(left_level>=right_level){
        return(4);
       	}else{
        return(2);
       }
    }else{
       if(left_level>=right_level){
        return(3);
       	}else{
        return(1);
        }
    }
}

    //　枠が正方形を通過しているか調べる
int quad_tree::search_dissection_line(point move){
   	if (x1<=move.x && move.x<x2 || y1<=move.y && move.y<y2 ){
         return (1);
    }else{
         return(0);
    }
}


   //　各値をセットする
void quad_tree::set(long xx1,long yy1,long xx2,long yy2,point p[], point move){
  	x1=xx1;y1=yy1;x2=xx2;y2=yy2;
    point_num=search_point_num(p);
    dissection_line=search_dissection_line(move);
    serach(x1,y1,x2,y2);
    type = search_type();
    type2 = search_type2();
}

long quad_tree::hash_table_size(void){
	return total_line_type*2;
};

void quad_tree::print_tree(void){
  	printf("x1=%d y1=%d x2=%d y2=%d type2=%d\n"
 		   ,x1,y1,x2,y2,type2);
  	printf("place[1]=%d place[2]=%d place[3]=%d place[4]=%d point_num=%d type=%d\n"
 		   ,place[1],place[2],place[3],place[4],point_num,search_type());
}







