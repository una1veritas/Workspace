
int compare_conection(conection e1[],conection e2[],long total1,long total2) {
	if(total1!=total2){
    	return 0;
	}
    for(long i=1;i<=total1;i++){
    	if(e1[i] == e2[i]){
        }else{
        	return 0;
        }
	}
    return 1;
}

int same_se(conection e1) {
	return((e1.s_line()==e1.e_line())&&(e1.s_number()==e1.e_number()));
}

void swap(conection *a,conection *b){
	conection tmp;
   	tmp=*a;
    *a=*b;
    *b=tmp;
}

void swap(conection2 *a,conection2 *b){
	conection2 tmp;
   	tmp=*a;
    *a=*b;
    *b=tmp;
}

void sort(conection a[],long n){
	long bound=n,t=1,i;
    while(t!=0){
    	t=0;
    	for(i=1;i<=bound-1;i++){
           	if(a[i]>a[i+1]){
        		swap(&a[i],&a[i+1]);
            	t=i;
            }
        }
        bound=t;
    }
}

void sort(conection2 a[],long n){
	long bound=n,t=1,i;
    while(t!=0){
    	t=0;
    	for(i=1;i<=bound-1;i++){
           	if(a[i]>a[i+1]){
        		swap(&a[i],&a[i+1]);
            	t=i;
            }
        }
        bound=t;
    }
}

void insert(conection a[],conection x,long *total){
	long i,j,tmp_total=*total,flag=0;
    x.sort();
    for(i=1;i<=tmp_total;i++){
    	if (a[i]>x){
        	for(j=tmp_total;j>=i;j--){
            	a[j+1]=a[j];
            }
            a[i]=x;
            tmp_total++;
        	*total=tmp_total;
            flag=1;
            break;
        }
    }
    if(flag==0){
	    tmp_total++;
   		*total=tmp_total;
    	a[tmp_total]=x;
    }
}

void del(conection a[],conection x,long *total){
	long i,j,tmp_total=*total;
    x.sort();

    for(i=1;i<=tmp_total-1;i++){
    	if(a[i]==x){
        	for(j=i;j<=tmp_total-1;j++){
        		a[j]=a[j+1];
            }
        	break;
        }
    }

    a[tmp_total].s_line(0);
    a[tmp_total].e_line(0);
    a[tmp_total].s_number(0);
    a[tmp_total].e_number(0);
    *total=tmp_total-1;
}

conection search(conection a[],long line,long number,long num ,long total){
	long i,j=0;
    for(i=1;i<=total;i++){
    	if(((a[i].s_line()==line)&&(a[i].s_number()==number))||((a[i].e_line()==line)&&(a[i].e_number()==number))){
            j++;
            if(j==num){
        		return a[i];
            }
        }
    }
    printf("line=%d number=%d\n",line,number);
    return a[0];
}

long search(conection a[],conection x,long total){
	long i;
    for(i=1;i<=total;i++){
    	if(a[i]==x){
       		return i;
        }
    }
    return 0;
}

long search(conection a[],conection x,long total,long b){
	long i;
    for(i=b;i<=total;i++){
    	if(a[i]==x){
       		return i;
        }
    }
    return 0;
}

long search_num(conection a[],long line,long number,long total){
	long i,ans=0;
    for(i=1;i<=total;i++){
    	if(((a[i].s_line()==line)&&(a[i].s_number()==number))||((a[i].e_line()==line)&&(a[i].e_number()==number))){
            ans++;
        }
    }
    return ans;
}

long srarch_setOfconection(setOfconection &  setcone,long quad_num,quad_tree quad[] ,sub_root **root){
  long i,offset,h;
  
  i=setcone.hash(quad[quad_num].hash_table_size());
  
   while(1){

    if((root[quad_num][i].cone==setcone)||(root[quad_num][i].length==-1)){
      sucsses++;
      return i;
    }
    fail++;
    i=(i+1)%quad[quad_num].hash_table_size();
   }
   
   fprintf(stderr,"Hash table overflow!\n");
   exit(1);
 }
 
void write_setOfconection(setOfconection & setcone,double length,long quad_num,quad_tree quad[] ,sub_root **root){
     long place,i;
     place=srarch_setOfconection(setcone,quad_num,quad ,root);

        root[quad_num][place].length=length;
        for(i=0;i<=2*r-1;i++){
	  root[quad_num][place].cone.set2(i,setcone.co(i));
	}
}

double read_setOfconection(setOfconection & setcone,long quad_num,quad_tree quad[] ,sub_root **root){
     long place;
     place=srarch_setOfconection(setcone,quad_num,quad ,root);
      return root[quad_num][place].length;
}

void write_length(setOfconection & setcone,double length,long quad_num,quad_tree quad[] ,sub_root **root){

	 write_setOfconection(setcone,length,quad_num,quad ,root);
}

double read_length(setOfconection setcone,long quad_num,quad_tree quad[] ,sub_root **root){
     setOfconection dummy;
        if(setcone==dummy){
			return 0.0;
        }else{
			return read_setOfconection(setcone,quad_num,quad,root);
        }
}
void write_conection(setOfconection & setcone,point_conection *p_cone,long quad_num,quad_tree quad[] ,sub_root **root){
    long place,i;
	place=srarch_setOfconection(setcone,quad_num,quad ,root);
    for(i=0;i<=quad[quad_num].point_num-1;i++){
        //printf("i=%d \n",i);
    	root[quad_num][place].p_cone[i]=p_cone[i];
    }
    	for(i=0;i<=2*r-1;i++){
	root[quad_num][place].cone.set2(i,setcone.co(i));
	}
}

void write_answer_factor(setOfconection & setcone,answer_factor2 & ans_fac,long quad_num,quad_tree quad[] ,sub_root **root){
  long place,i;
  place=srarch_setOfconection(setcone,quad_num,quad ,root);
  root[quad_num][place].length=ans_fac.length;
  for(i=0;i<=quad[quad_num].point_num-1;i++){
    root[quad_num][place].p_cone[i]=ans_fac.p_cone[i];
    
  }
  for(i=0;i<=2*r-1;i++){
    root[quad_num][place].cone.set2(i,setcone.co(i));
  }

}

answer_factor2 read_answer_factor(setOfconection & setcone,long quad_num,quad_tree quad[] ,sub_root **root){
     long place,i;
     answer_factor2 ans_fac;
     setOfconection dummy;
    if(setcone==dummy){
		ans_fac.length= 0.0;

    }else{
    	place=srarch_setOfconection(setcone,quad_num,quad ,root);
    	ans_fac.length = root[quad_num][place].length;
    	for(i=0;i<=quad[quad_num].point_num-1;i++){
	   	 	ans_fac.p_cone[i]=root[quad_num][place].p_cone[i];
        }
    }
     return ans_fac;
}












