
long random(long x){
  return (long)abs(rand() % x)+1;
}

void reverse(char s[]){
  int f,i,j;
  for(i=0,j=strlen(s)-1;i<j;i++,j--){
    f=s[i];
    s[i]=s[j];
    s[j]=f;
  }
}


void ltoa(long n,char s[]){
  
  long i,sign;
  if((sign = n)<0)
    n=-n;
  i=0;
  do{
    s[i++]=n%10 + '0';
  }while((n/=10)>0);
  
  if(sign<0)
    s[i++]='-';
  s[i]='\0';
  reverse(s);
  
}


    int log4(int x){
    	int tmp=0;
        while(x >= 4){
        	x /= 4;
            tmp++;
        }

        return tmp;
    }
    int log2(int x){
    	int tmp=0;
        while(x >= 2){
        	x /= 2;
            tmp++;
        }

        return tmp;
    }
long sum(long n,long start,long end){
	return n*(start+end)/2  ;
    }

long fact(long x){
	long ans=1;
	for(int i = 1;i<=x;i++){
     	ans *= i;
    }
    if(ans <= 0){
    	printf("fact‚ªŒvŽZ‚Å‚«‚Ü‚Ö‚ñ%d",x);
    	exit(1);
    }
    return ans;
}

long combination(long n, long i){
    long ans=1;
	if(n<=0){
    	return(0);
    }else if(i<=0){
    	return(0) ;
    }else if(i>n){
    	return(0);
    }else{
   for(long j = 0;j<=i-1;j++){
           	ans *=(n-j);
        }
     for(long j = 1;j<=i;j++){
           	ans /= j;
		}
    return ans;
    }
}

long H(long n,long i){
	return combination(n+i-1,i);
}

long permutation(long n,long i){
	long ans=1;
   for(long j = 0;j<=i-1;j++){
           	ans *=(n-j);
        }
  return ans;
    }

long sum_combination(long n,long s){
	long ans=0;
	if(s<=0){
    	return(0);
    }else{

    for(int i=1;i<=s;i++){
    	ans += combination(n,i);
    }
    return(ans);
    }
}

long sum_H(long n,long s){
	long ans=0;
	if(s<=0){
    	return(0);
    }else{

    	for(int i=1;i<=s;i++){
    		ans += H(n,i);
    	}
    	return(ans);
    }
}

long sum_permutation(long n,long s){
	long ans=0;
	if(s<=0){
    	return(0);
    }else{

    for(int i=1;i<=s;i++){
    	ans += permutation(n,i);
    }
    return(ans);
    }
}

long sum_combination2(long n,long s,long t){

	if(t>=s){
       return(sum_combination(n,s));
    }else{
       return(sum_combination(n,t));
    }
}
long sum_H2(long n,long s,long t){

	if(t>=s){
       return(sum_H(n,s));
    }else{
       return(sum_H(n,t));
    }
}

void return_edge(long x,long y,long *edge1,long *edge2){
	int s=1;
    while(y-sum(s,x-s+1,x)>0){
    s++;
    }
    *edge1=s;
    *edge2=y-sum(s-1,x-s+2,x)+s-1;
}

long return_num(long x,long edge1,long edge2){
	return sum(edge1-1,x-edge1+2,x)+edge2-edge1+1;
}

double distance(double x1,double y1,double x2,double y2){
	return sqrt(pow((double)labs((long int)(x1-x2)),2)+pow((double)labs((long int)(y1-y2)),2));
}

void swap(long *a,long *b){
long tmp;
	tmp=*a;
    *a=*b;
    *b=tmp;
}
void swap(short *a,short *b){
short tmp;
	tmp=*a;
    *a=*b;
    *b=tmp;
}

long search_line_num(long x,long total_line){
    long i=0;
    if(x==0){
       	return 0;
    }else{
		while(x>0){
    		x -=H(total_line,i+1);
        	i++;
    	}
        return i;
    }
}








