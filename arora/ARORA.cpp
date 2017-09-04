#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <values.h>
#include <iostream.h>
#include <string.h>
#include <time.h>

long Maxpoint=20;
long r=1;
long m=2;
long rand_flag=0;

double sucsses=0.0,fail =0.0;
long node_num=0;

long no_point_quad=0;

#define e 0.1   // soutai gosa
#define c 10
#define k  (long) fabs(log(2*8*c*Maxpoint/e)/log(2))+rand_flag
#define L (1L << k)
#define DoubleMax MAXDOUBLE
#define CommentOut(command)  /* command */  
#define Out_Answer_Display(command) /* command */    

#include "etc.h"
#include "arora.h"
#include "etc2.h"
answer_factor2 conection2length(long,setOfconection &,quad_tree*,sub_root**);



    //　与えられた正方枠を分割して木構造を作り出す
    //　また同じタイプのものは　ひとつの木で表す

void dissect(point p[],quad_tree quad[],point move,int type_table[],int *total){
 
int num=0,total_num=0;
    long x1,x2,y1,y2;
    long center_x,center_y;

    quad[0].set(0,0,L,L,p,move);
     while(num <= total_num){
       	if(quad[num].point_num > 1 ){

       		x1=quad[num].x1;x2=quad[num].x2;
       		y1=quad[num].y1;y2=quad[num].y2;
       		center_x=(x2-x1)/2+x1;
       		center_y=(y2-y1)/2+y1;

   			quad[num].place[1]=++total_num;
       		quad[total_num].set(x1,y1,center_x,center_y,p,move);
            if(quad[total_num].point_num == 0){
               	if(type_table[quad[total_num].type] == -1){
   	        		type_table[quad[total_num].type] = total_num;
       	        }else{
       				quad[num].place[1]=type_table[quad[total_num].type];
               	    total_num--;
   				}
            }

       		quad[num].place[2]=++total_num;
       		quad[total_num].set(x1,center_y,center_x,y2,p,move);
            if(quad[total_num].point_num == 0){
                if(type_table[quad[total_num].type] == -1){
   	        		type_table[quad[total_num].type] = total_num;
       	        }else{
       				quad[num].place[2]=type_table[quad[total_num].type];
               	    total_num--;
   				}
               }
       		quad[num].place[3]=++total_num;
       		quad[total_num].set(center_x,y1,x2,center_y,p,move);

            if(quad[total_num].point_num == 0){
   	            if(type_table[quad[total_num].type] == -1){
       	    		type_table[quad[total_num].type] = total_num;
           	    }else{
       				quad[num].place[3]=type_table[quad[total_num].type];
                   	total_num--;
   				}
             }

       		quad[num].place[4]=++total_num;
       		quad[total_num].set(center_x,center_y,x2,y2,p,move);

            if(quad[total_num].point_num == 0){
   	            if(type_table[quad[total_num].type] == -1){
       	    		type_table[quad[total_num].type] = total_num;
           	    }else{
       				quad[num].place[4]=type_table[quad[total_num].type];
                   	total_num--;
   				}
            }

	    /*	    printf("quad[%d] total_line_type=%d",num,quad[num].total_line_type);
	    quad[num].print_tree();
	    quad[num].print_lev();
	    printf("quad[%d] total_line(1,1)=%d %d\n",num,quad[num].total_line(1,1),quad[num].portals_num[1]);
	    quad[num].print();
	    */
	}
   		num++;
   	}
    *total = total_num;
}


/*正方形の辺ｍ１からｍ２をとおるすべて巡回路のうち
ｘ番目のものの本数num1とnum1本目の何番目かnum2を求める．*/
void search_line_num(long x,long total_line,long *num1,long *num2){
    long i=0;
    if(x==0){
       	*num1=0;
       	*num2=0;
    }else{
		while(x>0){
    		x -=H(total_line,i+1);
        	i++;
    	}
    		x +=H(total_line,i);
        	*num1=i;
        	*num2=x;
    }
}

/*正方形の辺ｍ１からｍ２をとおる巡回路のうち
ｘ番目（一本のときのみ）のもののエッジを求める．*/
conection search_edge(long x,int m1,int m2,quad_tree quad){
    conection ans;
    long edge1=1,tmp=quad.portals_num[m2],edge_2;

    if(m1==m2){
    	while(x>tmp){
    		tmp+=(quad.portals_num[m2]-edge1);
            edge1++;
        }
        tmp=sum(edge1-1,quad.portals_num[m2]-edge1+2,quad.portals_num[m2]) ;
        edge_2=x-tmp+edge1-1;
    }else{
      edge1 = (long) ceil( (double)x / (double)quad.portals_num[m2] );
        edge_2=x-(edge1-1)*quad.portals_num[m2];
    }
    ans.s_line(m1);
    ans.e_line(m2);

    ans.s_number(edge1);
    ans.e_number(edge_2);
    return ans;
}

void decide_start(long start[],long i,long d){
	if(start[i]+1 <= d){
    	start[i]++;
    }else{
    	decide_start(start,i-1,d);
        start[i]=start[i-1];
    }
}

long search_edge_num(long i,long j,long a ,long total_line){
	long tmp1=j;
    long start[2*r+2];
    long ans;
    long d=total_line;
    long s=d;

    for(long w=0;w<=i;w++){
    	start[w]=0;
    }
    while(tmp1>s){
    	decide_start(start,i,d);
        tmp1-=s;
        s=total_line-start[i];
	}
	start[i+1]=tmp1+start[i]-1;
    ans=start[a+1]+1;
	return ans;
}

void serach_min_length(quad_tree quad[],int total,sub_root **root,point p[]){
  conection tmp_edge;
  conection2 cone;
  point_conection min_cone;
  setOfconection setcone;
  answer_factor2 ans_fac;
  point tmp_point;
  long total_root=0,line_num=0;
  long i[11],l_num[11];
  long q,j,tmp,line[3];
  long t,s,w,v;
  double point[3];
  double tmp_length=0.0,tmp_length2=0.0,min_diff_length=DoubleMax,diff_length;
  double x[3],y[3];
  
  for(w=0;w<=total;w++){
    if(quad[w].point_num<=1){
      no_point_quad++;
      CommentOut(printf("w=%d\n",w);)
    	total_root=0;
	for( i[1]=0;i[1]<=sum_H(quad[w].total_line(1,1),r/2);i[1]++){
	  l_num[1]=search_line_num(i[1],quad[w].total_line(1,1));
	  for( i[2]=0;i[2]<=sum_H2(quad[w].total_line(1,2),r-l_num[1]*2,r);i[2]++){
	    l_num[2]=search_line_num(i[2],quad[w].total_line(1,2));
	    for( i[3]=0;i[3]<=sum_H2(quad[w].total_line(1,3),r-l_num[1]*2-l_num[2],r);i[3]++){
	      l_num[3]=search_line_num(i[3],quad[w].total_line(1,3));
	      for( i[4]=0;i[4]<=sum_H2(quad[w].total_line(1,4),r-l_num[1]*2-l_num[2]-l_num[3],r);i[4]++){
		l_num[4]=search_line_num(i[4],quad[w].total_line(1,4));
		for( i[5]=0;i[5]<=sum_H(quad[w].total_line(2,2),(r-l_num[2])/2);i[5]++){
		  l_num[5]=search_line_num(i[5],quad[w].total_line(2,2));
		  for( i[6]=0;i[6]<=sum_H2(quad[w].total_line(2,3),r-l_num[2]-l_num[5]*2,r-l_num[3]);i[6]++){
		    l_num[6]=search_line_num(i[6],quad[w].total_line(2,3));
		    for( i[7]=0;i[7]<=sum_H2(quad[w].total_line(2,4),r-l_num[2]-l_num[5]*2-l_num[6],r-l_num[4]);i[7]++){
		      l_num[7]=search_line_num(i[7],quad[w].total_line(2,4));
		      for( i[8]=0;i[8]<=sum_H(quad[w].total_line(3,3),(r-l_num[3]-l_num[6])/2);i[8]++){
			l_num[8]=search_line_num(i[8],quad[w].total_line(3,3));
			for( i[9]=0;i[9]<=sum_H2(quad[w].total_line(3,4),r-l_num[3]-l_num[6]-l_num[8]*2,r-l_num[4]-l_num[7]);i[9]++){
			  l_num[9]=search_line_num(i[9],quad[w].total_line(3,4));
			  for(  i[10]=0;i[10]<=sum_H(quad[w].total_line(4,4),(r-l_num[4]-l_num[7]-l_num[9])/2);i[10]++){
			    
			    setcone.init();
			    for(t=10;t>=1;t--){
			      search_line_num(i[t],quad[w].search_total_line_num(t),&q,&j);
			      
			      for(s=q;s>=1;s--){
        			tmp=search_edge_num(q,j,s,quad[w].search_total_line_num(t));
				return_edge(4,t,&line[1],&line[2]);
				tmp_edge=search_edge(tmp,line[1],line[2],quad[w]);
				cone.change_conection(tmp_edge);
				line_num++;
				setcone.set(line_num,cone);
				point[1]=quad[w].search_point(line[1],tmp_edge.s_number());
				point[2]=quad[w].search_point(line[2],tmp_edge.e_number());
				for(v=1;v<=2;v++){
				  if(line[v]==1){
				    x[v]=point[v];
				    y[v]=L;
				  }else if(line[v]==2){
				    x[v]=L;
				    y[v]=point[v];
				  }else if(line[v]==3){
				    x[v]=point[v];
				    y[v]=0;
				  }else if(line[v]==4){
				    x[v]=0;
				    y[v]=point[v];
				  }
				  
				}
				tmp_length2=distance(x[1],y[1],x[2],y[2]);
				tmp_length+=tmp_length2;
				if(quad[w].point_num==1){
				  tmp_point=quad[w].search_point_place(p);
				  diff_length=distance(x[1],y[1],tmp_point.x,tmp_point.y)
				    +distance(tmp_point.x,tmp_point.y,x[2],y[2])-tmp_length2;
				  if(min_diff_length>diff_length){
				    min_cone.line=cone;
				    min_cone.link=tmp_edge;
				    min_cone.set_line_num(1);
				    min_cone.point_set(quad[w].search_point_num2(p));
				    min_diff_length=diff_length;
				    
				  }
				}
				
			      }
			      
			    }
			    if(quad[w].point_num==1){
			      ans_fac.set( tmp_length+min_diff_length,&min_cone);
			      write_answer_factor(setcone,ans_fac,w,quad,root);
			    }else{
			      write_length(setcone,tmp_length,w,quad,root);
			    }
			    
			    tmp_length=0.0;
			    min_diff_length=DoubleMax;
			    
			    total_root++;
			    line_num=0;
			    
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
	
    }
  } 
  
}



//quad　a　の中のplaceiのline-numberはaではans＿line-ans_numberというのをだす．
edge serach_same_portals(edge x,int i,quad_tree a,quad_tree *quad){
  edge ans;
  long j,tmp_j;
  long line[2];
  int quad_type,half_flag[2];
  long half_portals;
  
  half_flag[0]=0;
  half_flag[1]=0;
  
  quad_type=a.type2;//quad[a.place[i]].search_type2();
  if((quad_type % 2)==1){
    if((i%2) ==1){
      line[0]=1;
    }else{
      line[0]=3;
    }
    if(i<=2){
      half_flag[0]=1;
    }
  }else{
    if((i%2) ==1){
      line[0]=3;
    }else{
      line[0]=1;
    }
    if(i>=3){
      half_flag[0]=1;
    }
  }
  
  if(quad_type<=2){
    if(i<=2){
      line[1]=2;
    }else{
      line[1]=4;
    }
    if((i%2)==1){
      half_flag[1]=1;
    }
  }else{
    if(i<=2){
      line[1]=4;
    }else{
      line[1]=2;
    }
    if((i%2)==0){
      half_flag[1]=1;
    }
  }
  
  for(j=0;j<=1;j++){
    if(a.level[line[j]]+1==quad[a.place[i]].level[x.line]){
      if(ans.line==0){
	ans.line=line[j];
      }else{
	ans.line=line[1-x.line%2];
      }
      tmp_j=j;
    }
  }
  
  if(x.number==0){
    ans.number=0;
  }else if(half_flag[tmp_j]==0){
    ans.number=x.number+(a.portals_num[ans.line]-quad[a.place[i]].portals_num[x.line]);
  }else{
    ans.number=quad[a.place[i]].portals_num[x.line]-x.number+1;
  }
  return ans;
}

long search_same_place(quad_tree a,edge x){
  long i,j;
  int flag=1,quad_type;
  
  if(x.line>=3){
    if(a.portals_num[x.line] / 2+1  <= x.number){
      flag=0;
    }
  }else{
    if((a.number[x.line]%2)==0){
      if(a.search_point(x.line,x.number) >= L/2 ){
	flag=0;
      }
    }else{
      
      if(a.search_point(x.line,x.number) > L/2 ){
	flag=0;
      }
    }
  }
  
  quad_type=a.type2;
  if(flag==1 && a.level[x.line]==0){
    return 4-quad_type+1;
  }else if(flag==0 && a.level[x.line]>0){
    return quad_type;
  }else if((x.line==4)||(x.line==1)){
    return (quad_type+1) % 4 +1;
  }else{
    return (6-quad_type) % 4 +1;
  }
  
}

//quad　a　のline-numberはaの中のplacei(=b)のans＿line-ans_numberというのをだす．
edge search_same_portals2(quad_tree a,edge x,quad_tree quad[]){
  long i,j;
  edge ans;
  int flag=1;
  if(x.line>=3){
    if(a.portals_num[x.line] / 2+1  <= x.number){
      flag=0;
    }
  }else{
    if(a.search_point(x.line,x.number) >= L/2 ){
      flag=0;
    }
  }
  
  i=search_same_place(a,x);
 
  for(j=1;j<=2;j++){
    if(quad[a.place[i]].level[j]==(a.level[x.line]+1)){
      ans.line=j;
      if(flag==0){
	ans.number=x.number-(a.portals_num[x.line]-quad[a.place[i]].portals_num[ans.line]);
      }else{
	ans.number= quad[a.place[i]].portals_num[ans.line]-x.number+1;
      }
      if((quad[a.place[i]].level[1]==quad[a.place[i]].level[2])){
	ans.line =2-(x.line % 2);
      }
      return ans;
    }
  }
  
}
//
long search_same_line(long line,quad_tree a,long place,quad_tree quad[]){
	edge in_edge,out_edge;
    in_edge.set(1,0) ;
	out_edge=serach_same_portals(in_edge,place,a,quad);
	if(out_edge.line==1 || out_edge.line==3){
    	return line;
    }else{
    	if(line==3){
        	return 4;
        }else if(line==4){
        	return 3;
        }else if(line==1){
        	return 2;
        }else {
        	return 1;
        }
    }
}
//aのplaceiのライン　input_lineはcross_numのどの線にあたるか？
long search_dissect_line(long place,long input_line,quad_tree a,quad_tree quad[]){
    long line=input_line;
   	line= search_same_line(input_line,a,place,quad);

    if(place==1){
    	if(line==3){
	    	return 0;
        }else{
    		return 2;
        }
    }else if(place==2){
    	if(line==3){
    		return 0;
        }else{
    		return 1;
        }
    }else if(place==3){
    	if(line==3){
    		return 3;
        }else{
    		return 2;
        }
    }else {
    	if(line==3){
    		return 3;
        }else{
	    	return 1;
        }
    }
}
long PossibleNextPortals_num(conection2 path,quad_tree a,long current,quad_tree quad[],long cross_num[]){
    long i,j,tmp,ans=0;
    edge end_edge;
    end_edge.set(path.e_line(),path.e_number());
    for(i=3;i<=4;i++){
    	if(cross_num[search_dissect_line(current,i,a,quad)]<r){
        	ans+=quad[a.place[current]].portals_num[i];
        }
    }

    if(search_same_place(a,end_edge)==current){
    	ans++;
    }
    node_num+=ans;
    return ans;
}

void PossibleNextPortals(conection2 path,quad_tree a,long current,quad_tree quad[],long cross_num[], edge *ans_edge){
    long i,j,tmp,total_ans=0;
    edge end_edge, tmp_edge;
    end_edge.set(path.e_line(),path.e_number());
    for(i=3;i<=4;i++){
    	if(cross_num[search_dissect_line(current,i,a,quad)]<r){
        	tmp_edge.line=i;
        	for(j=1;j<=quad[a.place[current]].portals_num[search_same_line(i,a,current,quad)];j++){
            	tmp_edge.number=j;
                ans_edge[total_ans]=tmp_edge;
                total_ans++;
            }
        }
    }
    if(search_same_place(a,end_edge)==current){
	    ans_edge[total_ans]=search_same_portals2(a,end_edge,quad);
    	total_ans++;
    }

}

long PossibleNextPortals_num0(long current,quad_tree quad[],long cross_num[]){
    long i,j,tmp,ans=0;
    quad_tree a=quad[0];
    for(i=3;i<=4;i++){
    	if(cross_num[search_dissect_line(current,i,a,quad)]<r){
        	ans+=quad[a.place[current]].portals_num[i];
        }
    }

    return ans;
}

void PossibleNextPortals0(long current,quad_tree quad[],long cross_num[], edge *ans_edge){
    long i,j,tmp,total_ans=0;
    edge tmp_edge;
    quad_tree a=quad[0];
    for(i=3;i<=4;i++){
    	if(cross_num[search_dissect_line(current,i,a,quad)]<r){
        	tmp_edge.line=i;
        	for(j=1;j<=quad[a.place[current]].portals_num[search_same_line(i,a,current,quad)];j++){
            	tmp_edge.number=j;
                ans_edge[total_ans]=tmp_edge;
                total_ans++;
            }
        }
    }
}


int search_next_place(long current,long dis_line){
	if(current==1){
	  if(dis_line==0){
	    return 2;
	  }else{
	    return 3;
	  }
	}else if(current==2){
	  if(dis_line==1){
	    return 4;
	  }else{
	    return 1;
	  }
	}else if(current==3){
	  if(dis_line==2){
	    return 1;
	  }else{
	    return 4;
	  }
	}else if(current==4){
	  if(dis_line==3){
        	return 3;
	  }else{
	    return 2;
	  }
	}

}

int serach_same_point(long total,long point_num,answer_factor2 & ans){
	long i;
    for(i=0;i<=total-1;i++){
    	if(ans.p_cone[i].c_point()==point_num){
        	return -1;
        }
    }
    return 0;
}


answer_factor2 dissect_conections(dissect_conection discone[], long total_conection ,long quad_num,setOfconection &  setcone ,quad_tree quad[],sub_root **root){
  // long tennokazu[2*r*4];
  //long tmp_tmp=0;

  long now_line_number,conection_type4_flag;
  long i,j,t;
  long set_place[4];
  long total_path,total_edge=0,total_line=0,start_place;
  edge now_edge,comp_edge,next_edge,end_edge;
  long change_cone_num,se_flag,path_num;
  long before_change_cone_num,before_path_num,before_se_flag;
  long next_point,before_point,conection_type,point_num;
  setOfconection dissect_cone[4];
  answer_factor2 answers[4],ans;
  ans.length=0;
  /*   
  for(i=0;i<=total_conection;i++){
    tennokazu[i]=0;
  }
  */
  for(i=0;i<=3;i++){
    set_place[i]=1;
  }
  for(i=0;i<=total_conection;i++){
    dissect_cone[discone[i].place-1].set(set_place[discone[i].place-1],discone[i]);
    set_place[discone[i].place-1]++;
  }

  for(i=0;i<=3;i++){
    dissect_cone[i].sort();
    //点のある正方形を通過しないとき
    if(quad[quad[quad_num].place[i+1]].point_num>=1){
      if(dissect_cone[i].element_num()==0){
	ans.length=DoubleMax;
	return ans;
      }
    }
  }
  for(i=0;i<=3;i++){
    answers[i]=conection2length(quad[quad_num].place[i+1],dissect_cone[i],quad,root);
    if(answers[i].length>=DoubleMax){
      ans.length=DoubleMax;
      return ans;
      
    }
    ans.length+=answers[i].length/2.0;
  }
  //ここかられんけつかいし     
 
  now_edge.set(discone[0].s_line(),discone[0].s_number());
  //連結本体
  for(i=0;i<=total_conection;i++){
    
    if(now_edge.line>=3){
      now_edge.line=search_same_line(now_edge.line,quad[quad_num],discone[i].place, quad);
    }
    
    total_path=total_edge/2+1;
    
    //いきなり端と端のとき
    
    if(discone[i].s_line()<=2 && discone[i].e_line()<=2){
      total_edge+=2;
      conection_type=1;
      now_edge.set(discone[i].s_line(),discone[i].s_number());
      end_edge.set(discone[i].e_line(),discone[i].e_number());
      
    }else if(discone[i].s_line()<=2 ){
      //かたっぽが端のとき，total_edgが偶数ならそれははじめのedge
      if(total_edge % 2 == 0){
	conection_type=2;
	change_cone_num=-1;
	before_change_cone_num=-1;
	now_edge.set(discone[i].s_line(),discone[i].s_number());
	end_edge.set(discone[i].e_line(),discone[i].e_number());
	next_edge.set(discone[i].e_line(),discone[i].e_number());
	next_edge.line=search_same_line(next_edge.line,quad[quad_num],discone[i].place, quad);
      }else{
	conection_type=3;
	now_edge.set(discone[i].e_line(),discone[i].e_number());
	end_edge.set(discone[i].s_line(),discone[i].s_number());
      }
      total_edge++;
    }else{
      //両方端でないとき，next_edgeは見た目のところのでいいので
      conection_type=4;
      comp_edge.set(discone[i].s_line(),discone[i].s_number());
      if(comp_edge==now_edge){
	end_edge.set(discone[i].e_line(),discone[i].e_number());
	next_edge.set(discone[i].e_line(),discone[i].e_number());
	next_edge.line=search_same_line(next_edge.line,quad[quad_num],discone[i].place, quad);
      }else{
	end_edge.set(discone[i].s_line(),discone[i].s_number());
	next_edge.set(discone[i].s_line(),discone[i].s_number());
	next_edge.line=search_same_line(next_edge.line,quad[quad_num],discone[i].place, quad);
      }
    }
    
    j=discone[i].place-1;//jはこの線のある場所
    point_num=0;now_line_number=0;conection_type4_flag=-1;
    for(t=0;t<= quad[quad[quad_num].place[j+1]].point_num-1;t++){
      if(discone[i]==answers[j].p_cone[t].line && 0==serach_same_point(total_line,answers[j].p_cone[t].c_point(),ans) &&
	((now_line_number==0)||(now_line_number==answers[j].p_cone[t].line_num())) ){
	
	//conection_type4_flag=0;

	//tennokazu[i]++;
	if(answers[j].p_cone[t].c_point()==0){
	  printf("answers[j].p_cone[t].c_point()=0\n");
	}
	if(now_line_number==0){
	  now_line_number= answers[j].p_cone[t].line_num();
	}
	
	point_num++;
	ans.p_cone[total_line].line=setcone.co2(total_path);
	ans.p_cone[total_line].link=answers[j].p_cone[t].link;
	ans.p_cone[total_line].set_line_num(total_path);
	ans.p_cone[total_line].point_set(answers[j].p_cone[t].c_point()) ;
	
	if(conection_type==2 || conection_type==4){

	  //後ろとリンクの用意
	  if(ans.p_cone[total_line].link.st()== end_edge && conection_type4_flag==-1){
	    //if(conection_type==2) {tmp_tmp+=5;}
	    //if(conection_type==4){ tmp_tmp++;}
	    next_point=ans.p_cone[total_line].c_point();
	    path_num=total_path;
	    change_cone_num=total_line;
	    se_flag=0;
	    if(now_edge==end_edge){
		conection_type4_flag=t*2+1;
	    }
	  }else if(ans.p_cone[total_line].link.en()==end_edge && conection_type4_flag==-1 ){
	    //if(conection_type==2) {tmp_tmp+=5;}
	    // if(conection_type==4) {tmp_tmp++;}
	    next_point=ans.p_cone[total_line].c_point();
	    path_num=total_path;
	    change_cone_num=total_line;
	    se_flag=1; 
	    if(now_edge==end_edge){
	      conection_type4_flag=t*2;
	    }
	  }
	}
	
	if(conection_type==4){
	  //前とつなぐ
	  if(before_change_cone_num==-1){  //前に点がないとき
	    if(ans.p_cone[total_line].link.st()==now_edge && conection_type4_flag!=t*2+1){
	      ans.p_cone[total_line].link.s_line(setcone.co2(total_path).s_line());
	      ans.p_cone[total_line].link.s_number(setcone.co2(total_path).s_number());
	    }else if( ans.p_cone[total_line].link.en()==now_edge && conection_type4_flag!=t*2){
	      ans.p_cone[total_line].link.e_line(setcone.co2(total_path).s_line());
	      ans.p_cone[total_line].link.e_number(setcone.co2(total_path).s_number());
	    }
	    
	  }else{
	    if(ans.p_cone[total_line].link.st()==now_edge && conection_type4_flag!=t*2+1 ){
	      //  tmp_tmp+=10;
	      if(before_se_flag==0){
		ans.p_cone[before_change_cone_num].link.s_line(0);
		ans.p_cone[before_change_cone_num].link.s_number(ans.p_cone[total_line].c_point());
	      }else{
		ans.p_cone[before_change_cone_num].link.e_line(0);
		ans.p_cone[before_change_cone_num].link.e_number(ans.p_cone[total_line].c_point());
		
	      }
	      ans.p_cone[total_line].link.s_line(0);
	      ans.p_cone[total_line].link.s_number(before_point);
	      
	    } else if(ans.p_cone[total_line].link.en()==now_edge && conection_type4_flag!=t*2){
	      //      tmp_tmp+=10;
	      if(before_se_flag==0){
		ans.p_cone[before_change_cone_num].link.s_line(0);
		ans.p_cone[before_change_cone_num].link.s_number(ans.p_cone[total_line].c_point());
	      }else{
		ans.p_cone[before_change_cone_num].link.e_line(0);
		ans.p_cone[before_change_cone_num].link.e_number(ans.p_cone[total_line].c_point());
		
	      }
	      ans.p_cone[total_line].link.e_line(0);
	      ans.p_cone[total_line].link.e_number(before_point);
	      
	    }
	  }
	}
	if(conection_type==3){
	  //前とつなぐ
	  if(before_change_cone_num==-1){
	    if( ans.p_cone[total_line].link.st()==now_edge){
	      ans.p_cone[total_line].link.s_line(setcone.co2(total_path).s_line());
	      ans.p_cone[total_line].link.s_number(setcone.co2(total_path).s_number());
	      if( ans.p_cone[total_line].link.en()==end_edge){
		ans.p_cone[total_line].link.e_line(setcone.co2(total_path).e_line());
		ans.p_cone[total_line].link.e_number(setcone.co2(total_path).e_number());
	      }
	    }else if( ans.p_cone[total_line].link.en()==now_edge){
	      ans.p_cone[total_line].link.e_line(setcone.co2(total_path).s_line());
	      ans.p_cone[total_line].link.e_number(setcone.co2(total_path).s_number());
	      if(ans.p_cone[total_line].link.st()==end_edge){
		ans.p_cone[total_line].link.s_line(setcone.co2(total_path).e_line());
		ans.p_cone[total_line].link.s_number(setcone.co2(total_path).e_number());
	      }
	    }else if(ans.p_cone[total_line].link.st()==end_edge){
	      ans.p_cone[total_line].link.s_line(setcone.co2(total_path).e_line());
	      ans.p_cone[total_line].link.s_number(setcone.co2(total_path).e_number());
	    }else if( ans.p_cone[total_line].link.en()==end_edge){
	      ans.p_cone[total_line].link.e_line(setcone.co2(total_path).e_line());
	      ans.p_cone[total_line].link.e_number(setcone.co2(total_path).e_number());
	    }
	    
	  }else{
	    if(ans.p_cone[total_line].link.st()==now_edge ){
	      if(before_se_flag==0){
		ans.p_cone[before_change_cone_num].link.s_line(0);
		ans.p_cone[before_change_cone_num].link.s_number(ans.p_cone[total_line].c_point());
	      }else{
		ans.p_cone[before_change_cone_num].link.e_line(0);
		ans.p_cone[before_change_cone_num].link.e_number(ans.p_cone[total_line].c_point());
		
	      }
	      ans.p_cone[total_line].link.s_line(0);
	      ans.p_cone[total_line].link.s_number(before_point);
	    } else if(ans.p_cone[total_line].link.en()==now_edge){
	      if(before_se_flag==0){
		ans.p_cone[before_change_cone_num].link.s_line(0);
		ans.p_cone[before_change_cone_num].link.s_number(ans.p_cone[total_line].c_point());
	      }else{
		ans.p_cone[before_change_cone_num].link.e_line(0);
		ans.p_cone[before_change_cone_num].link.e_number(ans.p_cone[total_line].c_point());
		
	      }
	      ans.p_cone[total_line].link.e_line(0);
	      ans.p_cone[total_line].link.e_number(before_point);
	    }
	    
	    if(ans.p_cone[total_line].link.st()==end_edge){
	      ans.p_cone[total_line].link.s_line(setcone.co2(total_path).e_line());
	      ans.p_cone[total_line].link.s_number(setcone.co2(total_path).e_number());
	    }else if( ans.p_cone[total_line].link.en()==end_edge){
	      ans.p_cone[total_line].link.e_line(setcone.co2(total_path).e_line());
	      ans.p_cone[total_line].link.e_number(setcone.co2(total_path).e_number());
	    }
	  }
	}
	
	if(conection_type==2){
	  if( ans.p_cone[total_line].link.st()==now_edge){
	    ans.p_cone[total_line].link.s_line(setcone.co2(total_path).s_line());
	    ans.p_cone[total_line].link.s_number(setcone.co2(total_path).s_number());
	  }else if( ans.p_cone[total_line].link.en()==now_edge){
	    ans.p_cone[total_line].link.e_line(setcone.co2(total_path).s_line());
	    ans.p_cone[total_line].link.e_number(setcone.co2(total_path).s_number());
	  }
	}
	
	if(conection_type==1){
	  if( ans.p_cone[total_line].link.st()==now_edge){
	    ans.p_cone[total_line].link.s_line(setcone.co2(total_path).s_line());
	    ans.p_cone[total_line].link.s_number(setcone.co2(total_path).s_number());
	    if( ans.p_cone[total_line].link.en()==end_edge){
	      ans.p_cone[total_line].link.e_line(setcone.co2(total_path).e_line());
	      ans.p_cone[total_line].link.e_number(setcone.co2(total_path).e_number());
	    }
	  }else if( ans.p_cone[total_line].link.en()==now_edge){
	    ans.p_cone[total_line].link.e_line(setcone.co2(total_path).s_line());
	    ans.p_cone[total_line].link.e_number(setcone.co2(total_path).s_number());
	    if(ans.p_cone[total_line].link.st()==end_edge){
	      ans.p_cone[total_line].link.s_line(setcone.co2(total_path).e_line());
	      ans.p_cone[total_line].link.s_number(setcone.co2(total_path).e_number());
	    }
	  }else if(ans.p_cone[total_line].link.st()==end_edge){
	    ans.p_cone[total_line].link.s_line(setcone.co2(total_path).e_line());
	    ans.p_cone[total_line].link.s_number(setcone.co2(total_path).e_number());
	    if( ans.p_cone[total_line].link.en()==now_edge){
	      ans.p_cone[total_line].link.e_line(setcone.co2(total_path).s_line());
	      ans.p_cone[total_line].link.e_number(setcone.co2(total_path).s_number());
	    }
	  }else if( ans.p_cone[total_line].link.en()==end_edge){
	    ans.p_cone[total_line].link.e_line(setcone.co2(total_path).e_line());
	    ans.p_cone[total_line].link.e_number(setcone.co2(total_path).e_number());
	    if( ans.p_cone[total_line].link.st()==now_edge){
	      ans.p_cone[total_line].link.s_line(setcone.co2(total_path).s_line());
	      ans.p_cone[total_line].link.s_number(setcone.co2(total_path).s_number());
	    }
	  }
	}
	total_line++;
	
      }
    }
    
    before_change_cone_num=change_cone_num;
    before_se_flag=se_flag;
    before_point=next_point;
    before_path_num=path_num;
    now_edge=next_edge;

    if(conection_type==3 && point_num ==0 && before_change_cone_num!=-1){//next_edge.line<=2 && start_flag==1 && edge_edge_flag==0){
      if(before_se_flag==0){
	ans.p_cone[before_change_cone_num].link.s_line(setcone.co2(before_path_num).e_line());
	ans.p_cone[before_change_cone_num].link.s_number(setcone.co2(before_path_num).e_number());
	ans.p_cone[before_change_cone_num].link.sort();
      }else{
	ans.p_cone[before_change_cone_num].link.e_line(setcone.co2(before_path_num).e_line());
	ans.p_cone[before_change_cone_num].link.e_number(setcone.co2(before_path_num).e_number());
	ans.p_cone[before_change_cone_num].link.sort();
      }
    }
  }
  /*
  i=-1;j=0;
  for(t=0;t<= quad[quad_num].point_num-1;t++){
    
    if(ans.p_cone[t].line_num()!=i){
      j=0;
      i=ans.p_cone[t].line_num();

    }
    
    if(ans.p_cone[t].link.s_line()>0){
      j++;
    }
    if(ans.p_cone[t].link.e_line()>0){
      j++;
    }
    if(j==4 && quad[quad_num].point_num<=4){
      printf("error j=%d quad_num=%d tmp_tmp=%d \n",j,quad_num,tmp_tmp);
      for(long w=0;w<=total_conection;w++){  
	printf("place=%d tennokazu=%d disconect=",discone[w].place,tennokazu[w]);
	discone[w].print3();
      }
      ans.print(quad[quad_num].point_num);
    }

  } */
  return ans;
}

answer_factor2 dissect_conections0(edge start_edge,dissect_conection discone[], long total_conection ,long quad_num,setOfconection & setcone ,quad_tree quad[],sub_root **root){
  long now_line_number,conection_type4_flag;
  long i,j,t;
  long set_place[4];
  long total_path,total_edge=0,total_line=0,start_place;
  edge now_edge,comp_edge,next_edge,end_edge;
  long change_cone_num,se_flag,path_num;
  long before_change_cone_num,before_path_num,before_se_flag;
  long next_point,before_point,conection_type,point_num;
  setOfconection dissect_cone[4];
  answer_factor2 answers[4],ans;
  ans.length=0;
  
  for(i=0;i<=3;i++){
    set_place[i]=1;
  }
  for(i=0;i<=total_conection;i++){
    dissect_cone[discone[i].place-1].set(set_place[discone[i].place-1],discone[i]);
    set_place[discone[i].place-1]++;
  }
  for(i=0;i<=3;i++){
    dissect_cone[i].sort();
    
    //点のある正方形を通過しないとき
    if(quad[quad[quad_num].place[i+1]].point_num>=1){
      if(dissect_cone[i].element_num()==0){
	ans.length=DoubleMax;
	return ans;
      }
    }
  }
  for(i=0;i<=3;i++){
    answers[i]=conection2length(quad[quad_num].place[i+1],dissect_cone[i],quad,root);
    if(answers[i].length>=DoubleMax){
      ans.length=DoubleMax;
      return ans;
      
    }
    ans.length+=answers[i].length/2.0;
  }
  change_cone_num=-1;
  before_change_cone_num=-1;
  now_edge=start_edge ;

  for(i=0;i<=total_conection;i++){
    if(now_edge.line>=3){
      now_edge.line=search_same_line(now_edge.line,quad[quad_num],discone[i].place, quad);
    }
    
    total_path=total_edge/2+1;
    //いきなり端と端のとき
    
    if(discone[i].s_line()<=2 && discone[i].e_line()<=2){
      total_edge+=2;
      conection_type=1;
      now_edge.set(discone[i].s_line(),discone[i].s_number());
      end_edge.set(discone[i].e_line(),discone[i].e_number());
      
    }else if(discone[i].s_line()<=2 ){
      //かたっぽが端のとき，total_edgが偶数ならそれははじめのedge
      if(total_edge % 2 == 0){
	conection_type=2;
	change_cone_num=-1;
	now_edge.set(discone[i].s_line(),discone[i].s_number());
	end_edge.set(discone[i].e_line(),discone[i].e_number());
	next_edge.set(discone[i].e_line(),discone[i].e_number());
	next_edge.line=search_same_line(next_edge.line,quad[quad_num],discone[i].place, quad);
      }else{
	conection_type=3;
	now_edge.set(discone[i].e_line(),discone[i].e_number());
	end_edge.set(discone[i].s_line(),discone[i].s_number());
            }
      total_edge++;
    }else{
      //両方端でないとき，next_edgeは見た目のところのでいいので
        //前回のnext_edgeの値が変化していないのに着目して
      //next_edgeでないほうのedgeをnext_edgeにする．
      conection_type=4;
      comp_edge.set(discone[i].s_line(),discone[i].s_number());
      if(comp_edge==now_edge){
            	end_edge.set(discone[i].e_line(),discone[i].e_number());
            	next_edge.set(discone[i].e_line(),discone[i].e_number());
                next_edge.line=search_same_line(next_edge.line,quad[quad_num],discone[i].place, quad);
      }else{
	end_edge.set(discone[i].s_line(),discone[i].s_number());
            	next_edge.set(discone[i].s_line(),discone[i].s_number());
                next_edge.line=search_same_line(next_edge.line,quad[quad_num],discone[i].place, quad);
      }
    }
    j=discone[i].place-1;
    point_num=0;now_line_number=0;conection_type4_flag=-1;
    for(t=0;t<= quad[quad[quad_num].place[j+1]].point_num-1;t++){
      if(discone[i]==answers[j].p_cone[t].line && 0==serach_same_point(total_line,answers[j].p_cone[t].c_point(),ans) &&
	 ((now_line_number==0)||(now_line_number==answers[j].p_cone[t].line_num())) ){
	
	if(now_line_number==0){
	  now_line_number= answers[j].p_cone[t].line_num();
	}
	
	point_num++;
	ans.p_cone[total_line].line=setcone.co2(total_path);
	ans.p_cone[total_line].link=answers[j].p_cone[t].link;
	ans.p_cone[total_line].set_line_num(total_path);
	ans.p_cone[total_line].point_set(answers[j].p_cone[t].c_point()) ;
	
	if(conection_type==2 || conection_type==4){
	  //後ろとリンクの用意
	  if(ans.p_cone[total_line].link.st()== end_edge && conection_type4_flag==-1){
	    next_point=ans.p_cone[total_line].c_point();
	    path_num=total_path;
	    change_cone_num=total_line;
	    se_flag=0;
	    if(now_edge==end_edge){
		conection_type4_flag=t*2+1;
	    }
	  }else if(ans.p_cone[total_line].link.en()==end_edge && conection_type4_flag==-1){
	    next_point=ans.p_cone[total_line].c_point();
	    path_num=total_path;
	    change_cone_num=total_line;
	    se_flag=1; 
	    if(now_edge==end_edge){
	      conection_type4_flag=t*2;
	    }
	  }
	}
	
	if(conection_type==4){
	  //前とつなぐ
	  if(before_change_cone_num==-1){  //前に点がないとき
	    if( ans.p_cone[total_line].link.st()==now_edge && conection_type4_flag!=t*2+1){
	      ans.p_cone[total_line].link.s_line(setcone.co2(total_path).s_line());
	      ans.p_cone[total_line].link.s_number(setcone.co2(total_path).s_number());
	    }else if( ans.p_cone[total_line].link.en()==now_edge && conection_type4_flag!=t*2){
	      ans.p_cone[total_line].link.e_line(setcone.co2(total_path).s_line());
	      ans.p_cone[total_line].link.e_number(setcone.co2(total_path).s_number());
	    }
	    
	  }else{
	    if(ans.p_cone[total_line].link.st()==now_edge && conection_type4_flag!=t*2+1 ){
	      if(before_se_flag==0){
		ans.p_cone[before_change_cone_num].link.s_line(0);
		ans.p_cone[before_change_cone_num].link.s_number(ans.p_cone[total_line].c_point());
	      }else{
		ans.p_cone[before_change_cone_num].link.e_line(0);
		ans.p_cone[before_change_cone_num].link.e_number(ans.p_cone[total_line].c_point());
		
	      }
	      ans.p_cone[total_line].link.s_line(0);
	      ans.p_cone[total_line].link.s_number(before_point);
	      
	    } else if(ans.p_cone[total_line].link.en()==now_edge && conection_type4_flag!=t*2){
	      if(before_se_flag==0){
		ans.p_cone[before_change_cone_num].link.s_line(0);
		ans.p_cone[before_change_cone_num].link.s_number(ans.p_cone[total_line].c_point());
	      }else{
		ans.p_cone[before_change_cone_num].link.e_line(0);
		ans.p_cone[before_change_cone_num].link.e_number(ans.p_cone[total_line].c_point());
		
	      }
	      ans.p_cone[total_line].link.e_line(0);
	      ans.p_cone[total_line].link.e_number(before_point);
	      
	    }
	  }
	}
	/*
       	if(conection_type==2 || conection_type==4){
	  //後ろとリンクの用意
	  if(ans.p_cone[total_line].link.st()== end_edge ){
	    next_point=ans.p_cone[total_line].c_point();
	    path_num=total_path;
	    change_cone_num=total_line;
	    se_flag=0;
	  }else if(ans.p_cone[total_line].link.en()==end_edge){
	    next_point=ans.p_cone[total_line].c_point();
	    path_num=total_path;
	    change_cone_num=total_line;
	    se_flag=1;
	  }
	}
	
	if(conection_type==4){
	  //前とつなぐ
	  if(before_change_cone_num==-1){
	    if( ans.p_cone[total_line].link.st()==now_edge){
	      ans.p_cone[total_line].link.s_line(setcone.co2(total_path).s_line());
	      ans.p_cone[total_line].link.s_number(setcone.co2(total_path).s_number());
	    }else if( ans.p_cone[total_line].link.en()==now_edge){
	      ans.p_cone[total_line].link.e_line(setcone.co2(total_path).s_line());
	      ans.p_cone[total_line].link.e_number(setcone.co2(total_path).s_number());
	    }
	    
	  }else{
	    if(ans.p_cone[total_line].link.st()==now_edge ){
	      if(before_se_flag==0){
		ans.p_cone[before_change_cone_num].link.s_line(0);
		ans.p_cone[before_change_cone_num].link.s_number(ans.p_cone[total_line].c_point());
	      }else{
		ans.p_cone[before_change_cone_num].link.e_line(0);
		ans.p_cone[before_change_cone_num].link.e_number(ans.p_cone[total_line].c_point());
		
	      }
	      ans.p_cone[total_line].link.s_line(0);
	      ans.p_cone[total_line].link.s_number(before_point);
	      
	    } else if(ans.p_cone[total_line].link.en()==now_edge){
	      if(before_se_flag==0){
		ans.p_cone[before_change_cone_num].link.s_line(0);
		ans.p_cone[before_change_cone_num].link.s_number(ans.p_cone[total_line].c_point());
	      }else{
		ans.p_cone[before_change_cone_num].link.e_line(0);
		ans.p_cone[before_change_cone_num].link.e_number(ans.p_cone[total_line].c_point());
		
	      }
	      ans.p_cone[total_line].link.e_line(0);
	      ans.p_cone[total_line].link.e_number(before_point);
	      
	    }
	  }
	} 
	*/
	if(conection_type==3){
	  //前とつなぐ
	  if(before_change_cone_num==-1){
	    if( ans.p_cone[total_line].link.st()==now_edge){
	      ans.p_cone[total_line].link.s_line(setcone.co2(total_path).s_line());
	      ans.p_cone[total_line].link.s_number(setcone.co2(total_path).s_number());
	      if( ans.p_cone[total_line].link.en()==end_edge){
		ans.p_cone[total_line].link.e_line(setcone.co2(total_path).e_line());
		ans.p_cone[total_line].link.e_number(setcone.co2(total_path).e_number());
	      }
	    }else if( ans.p_cone[total_line].link.en()==now_edge){
	      ans.p_cone[total_line].link.e_line(setcone.co2(total_path).s_line());
	      ans.p_cone[total_line].link.e_number(setcone.co2(total_path).s_number());
	      if(ans.p_cone[total_line].link.st()==end_edge){
		ans.p_cone[total_line].link.s_line(setcone.co2(total_path).e_line());
		ans.p_cone[total_line].link.s_number(setcone.co2(total_path).e_number());
	      }
	    }else if(ans.p_cone[total_line].link.st()==end_edge){
	      ans.p_cone[total_line].link.s_line(setcone.co2(total_path).e_line());
	      ans.p_cone[total_line].link.s_number(setcone.co2(total_path).e_number());
	    }else if( ans.p_cone[total_line].link.en()==end_edge){
	      ans.p_cone[total_line].link.e_line(setcone.co2(total_path).e_line());
	      ans.p_cone[total_line].link.e_number(setcone.co2(total_path).e_number());
	    }
	    
	  }else{
	    if(ans.p_cone[total_line].link.st()==now_edge ){
	      if(before_se_flag==0){
		ans.p_cone[before_change_cone_num].link.s_line(0);
		ans.p_cone[before_change_cone_num].link.s_number(ans.p_cone[total_line].c_point());
	      }else{
		ans.p_cone[before_change_cone_num].link.e_line(0);
		ans.p_cone[before_change_cone_num].link.e_number(ans.p_cone[total_line].c_point());
		
	      }
	      ans.p_cone[total_line].link.s_line(0);
	      ans.p_cone[total_line].link.s_number(before_point);
	    } else if(ans.p_cone[total_line].link.en()==now_edge){
	      if(before_se_flag==0){
		ans.p_cone[before_change_cone_num].link.s_line(0);
		ans.p_cone[before_change_cone_num].link.s_number(ans.p_cone[total_line].c_point());
	      }else{
		ans.p_cone[before_change_cone_num].link.e_line(0);
		ans.p_cone[before_change_cone_num].link.e_number(ans.p_cone[total_line].c_point());
		
	      }
	      ans.p_cone[total_line].link.e_line(0);
	      ans.p_cone[total_line].link.e_number(before_point);
	    }
	    
	    if(ans.p_cone[total_line].link.st()==end_edge){
	      ans.p_cone[total_line].link.s_line(setcone.co2(total_path).e_line());
	      ans.p_cone[total_line].link.s_number(setcone.co2(total_path).e_number());
	    }else if( ans.p_cone[total_line].link.en()==end_edge){
	      ans.p_cone[total_line].link.e_line(setcone.co2(total_path).e_line());
	      ans.p_cone[total_line].link.e_number(setcone.co2(total_path).e_number());
	    }
	  }
	}
	
	if(conection_type==2){
	  if( ans.p_cone[total_line].link.st()==now_edge){
	    ans.p_cone[total_line].link.s_line(setcone.co2(total_path).s_line());
	    ans.p_cone[total_line].link.s_number(setcone.co2(total_path).s_number());
	  }else if( ans.p_cone[total_line].link.en()==now_edge){
	    ans.p_cone[total_line].link.e_line(setcone.co2(total_path).s_line());
	    ans.p_cone[total_line].link.e_number(setcone.co2(total_path).s_number());
	  }
	}
	
	if(conection_type==1){
	  if( ans.p_cone[total_line].link.st()==now_edge){
	    ans.p_cone[total_line].link.s_line(setcone.co2(total_path).s_line());
	    ans.p_cone[total_line].link.s_number(setcone.co2(total_path).s_number());
	    if( ans.p_cone[total_line].link.en()==end_edge){
	      ans.p_cone[total_line].link.e_line(setcone.co2(total_path).e_line());
	      ans.p_cone[total_line].link.e_number(setcone.co2(total_path).e_number());
	    }
	  }else if( ans.p_cone[total_line].link.en()==now_edge){
	    ans.p_cone[total_line].link.e_line(setcone.co2(total_path).s_line());
	    ans.p_cone[total_line].link.e_number(setcone.co2(total_path).s_number());
	    if(ans.p_cone[total_line].link.st()==end_edge){
	      ans.p_cone[total_line].link.s_line(setcone.co2(total_path).e_line());
	      ans.p_cone[total_line].link.s_number(setcone.co2(total_path).e_number());
	    }
	  }else if(ans.p_cone[total_line].link.st()==end_edge){
	    ans.p_cone[total_line].link.s_line(setcone.co2(total_path).e_line());
	    ans.p_cone[total_line].link.s_number(setcone.co2(total_path).e_number());
	    if( ans.p_cone[total_line].link.en()==now_edge){
	      ans.p_cone[total_line].link.e_line(setcone.co2(total_path).s_line());
	      ans.p_cone[total_line].link.e_number(setcone.co2(total_path).s_number());
	    }
	  }else if( ans.p_cone[total_line].link.en()==end_edge){
	    ans.p_cone[total_line].link.e_line(setcone.co2(total_path).e_line());
	    ans.p_cone[total_line].link.e_number(setcone.co2(total_path).e_number());
	    if( ans.p_cone[total_line].link.st()==now_edge){
	      ans.p_cone[total_line].link.s_line(setcone.co2(total_path).s_line());
	      ans.p_cone[total_line].link.s_number(setcone.co2(total_path).s_number());
	    }
	  }
	}
	total_line++;
	
      } 
    }
    before_change_cone_num=change_cone_num;
    before_se_flag=se_flag;
    before_point=next_point;
    before_path_num=path_num;
    now_edge=next_edge;
    
  }
  if(before_change_cone_num!=-1){//next_edge.line<=2 && start_flag==1 && edge_edge_flag==0){
    if(before_se_flag==0){
      ans.p_cone[before_change_cone_num].link.s_line(start_edge.line);
      ans.p_cone[before_change_cone_num].link.s_number(start_edge.number);
                ans.p_cone[before_change_cone_num].link.sort();
    }else{
      ans.p_cone[before_change_cone_num].link.e_line(start_edge.line);
      ans.p_cone[before_change_cone_num].link.e_number(start_edge.number);
      ans.p_cone[before_change_cone_num].link.sort();
    }
  }
  return ans;
}


answer_factor2 search_length(edge start_edge,int place,long total_conection,long total_path,long cross_num[],long quad_num,setOfconection &  setcone,quad_tree quad[],sub_root **root,dissect_conection discone[]){
    long j,possible_num;
    answer_factor2 ans,tmp_ans;
    edge tmp_edge,end_edge,tmp_end_edge;
    edge next_portals[2*m+1],next_edge;
    conection tmp_conection;
    int dissect_line,next_place,end_place;
    ans.length=DoubleMax;
    tmp_end_edge.set(setcone.co2(total_path).e_line(),setcone.co2(total_path).e_number());
    end_place=search_same_place(quad[quad_num],tmp_end_edge);
    end_edge=search_same_portals2(quad[quad_num],tmp_end_edge,quad) ;

    if(start_edge.line>=3){
    	start_edge.line=search_same_line(start_edge.line,quad[quad_num],place, quad);
    }
    possible_num=PossibleNextPortals_num(setcone.co2(total_path),quad[quad_num],place,quad,cross_num);
    PossibleNextPortals(setcone.co2(total_path),quad[quad_num],place,quad,cross_num,next_portals);
    for(j=0;j<=possible_num-1;j++){
      // next_portals[j].print();
        if(next_portals[j].line>=3){
         // 移動が内側だったとき
            //どちら向きに移動するかnext_edgeでだす
    		next_edge.line=search_same_line(next_portals[j].line,quad[quad_num],place, quad);
    		next_edge.number=next_portals[j].number;
    	    dissect_line=search_dissect_line(place,next_portals[j].line,quad[quad_num],quad);
			cross_num[dissect_line]++;
    	    next_place=search_next_place(place,dissect_line);
       		tmp_conection.set(start_edge.line,start_edge.number,next_portals[j].line,next_portals[j].number);

			discone[total_conection].change_conection(tmp_conection);
    	    discone[total_conection].place=place;


			tmp_ans=search_length(next_edge,next_place,total_conection+1,total_path,cross_num,quad_num,setcone,quad,root,discone);
            if(tmp_ans<ans){
               	ans=tmp_ans;
            }

			cross_num[dissect_line]--;
			discone[total_conection].init();
       		discone[total_conection].place=0;

    	}else{
        	//移動が外側だったとき
           	//next_edge=next_portals[j];
	       		tmp_conection.set(start_edge.line,start_edge.number,next_portals[j].line,next_portals[j].number);
				discone[total_conection].change_conection(tmp_conection);
    	    	discone[total_conection].place=place;

            if(total_path!=setcone.element_num()){
		    	next_edge.set(setcone.co2(total_path+1).s_line(),setcone.co2(total_path+1).s_number());
            	next_place=search_same_place(quad[quad_num],next_edge);
   				next_edge=search_same_portals2(quad[quad_num],next_edge,quad);


				tmp_ans=search_length(next_edge,next_place,total_conection+1,total_path+1,cross_num,quad_num,setcone,quad,root,discone);
                if(tmp_ans<ans){
                	ans=tmp_ans;
                }
				discone[total_conection].init();
		        discone[total_conection].place=0;
            }else{

                tmp_ans=dissect_conections(discone,total_conection ,quad_num,setcone,quad,root);
                if(tmp_ans<ans){
                	ans=tmp_ans;
                }
            }
        }

    }
    return ans;
}

answer_factor2 search_length0(edge start_edge,int place,edge end_edge,int end_place,long total_conection,long cross_num[],quad_tree quad[],sub_root **root,dissect_conection discone[]){
  long j,possible_num,quad_num=0;
  answer_factor2 ans,tmp_ans;
  edge tmp_edge,tmp_end_edge;
  edge next_portals[2*m+1],next_edge;
  conection tmp_conection;
  setOfconection dummy_cone;
  conection2 dummy_cone2;
  int dissect_line,next_place;
  
  ans.length=DoubleMax;

  if(start_edge.line>=3){
    start_edge.line=search_same_line(start_edge.line,quad[quad_num],place, quad);
  }
  possible_num=PossibleNextPortals_num0(place,quad,cross_num);
  
  PossibleNextPortals0(place,quad,cross_num,next_portals);

  for(j=0;j<=possible_num-1;j++){
    //next_portals[j].print();
    next_edge.line=search_same_line(next_portals[j].line,quad[quad_num],place, quad);
    next_edge.number=next_portals[j].number;
    
    dissect_line=search_dissect_line(place,next_portals[j].line,quad[quad_num],quad);
    cross_num[dissect_line]++;
    next_place=search_next_place(place,dissect_line);
    tmp_conection.set(start_edge.line,start_edge.number,next_portals[j].line,next_portals[j].number);
    
    discone[total_conection].change_conection(tmp_conection);
    discone[total_conection].place=place;
    
    if((next_portals[j]==end_edge && place==end_place)||(next_edge==end_edge && next_place==end_place)){
      CommentOut(   printf("==========\n");
		    for(long i=0 ;i<=total_conection;i++){
		      printf("place=%d disconect=",discone[i].place);
		      discone[i].print2();
		    } )
	dummy_cone2.set(end_edge.line,end_edge.number,end_edge.line,end_edge.number);
	dummy_cone.set(1,dummy_cone2);
	tmp_ans=dissect_conections0(end_edge,discone,total_conection ,0,dummy_cone,quad,root);
	if(tmp_ans<ans){
	  ans=tmp_ans;
	  //printf("This root is shortest.\n");
	}
	
    }else{
			tmp_ans=search_length0(next_edge,next_place,end_edge,end_place,total_conection+1,cross_num,quad,root,discone);
			if(tmp_ans<ans){
			  ans=tmp_ans;
			}
    }
    
			cross_num[dissect_line]--;
			discone[total_conection].init();
			discone[total_conection].place=0;
			
			
			
    }
  return ans;
}

answer_factor2 conection2length(long quad_num,setOfconection & setcone,quad_tree quad[],sub_root **root){
  double length;
  answer_factor2 ans;
  long j;
  int place;
  conection dummy;
  dissect_conection discone[2*r*4];
  edge start_edge;
  long cross_num[4];
  for(j=0;j<=3;j++){
    cross_num[j]=0;
  }
  
  ans=read_answer_factor(setcone,quad_num,quad,root);
  if(ans.length != -1.0){
    return ans;
  }else{
    
    start_edge.set(setcone.co2(1).s_line(),setcone.co2(1).s_number());
    place=search_same_place(quad[quad_num],start_edge);
    start_edge=search_same_portals2(quad[quad_num],start_edge,quad);
    ans=search_length(start_edge,place,0,1,cross_num,quad_num,setcone,quad,root,discone);
    write_answer_factor(setcone,ans,quad_num,quad,root);
    
    return ans;
  }
}

answer_factor2 search_minimum(quad_tree quad[],sub_root **root){
  
  answer_factor2 ans,tmp_ans;
  long i,j,t,place;
  dissect_conection discone[2*r*4];
  edge start_edge;
  long cross_num[4];
  
  ans.length=DoubleMax;
  

  for(place=1;place<=4;place++){
    for(i=3;i<=4;i++){
      for(j=1;j<=m;j++){
	start_edge.set(i,j);
	for(t=0;t<=3;t++){
	  cross_num[t]=0;
	}
	for(t=0;t<=2*r*4-1;t++){
	  discone[t].init2();
	}

	tmp_ans=search_length0(start_edge,place,start_edge,place,0,cross_num,quad,root,discone);
	if(tmp_ans<ans){
	  ans=tmp_ans;
	}
      }
    }
  }
  return ans;
  
}

   //　点の初期化
void init_point(point *p){
  long i,tmp;
  long div=8;
  if(rand_flag==1){
    div=16;
  }

  for(i=1;i<=Maxpoint;i++){
    tmp = random((long)L/div-1);
    p[i].x= tmp * 8;
    tmp = random((long)L/div-1);
    p[i].y= tmp * 8;
    CommentOut(printf("%d x=%d y=%d\n",i,p[i].x,p[i].y);)
  }
}

void arora(char *input_file,char *output_file){
    point p[Maxpoint+1];
    point ans_p[Maxpoint+1];
    long ans_p_num[Maxpoint+1];
    long move_diff;
    double ans_length=0.0;
    quad_tree quad[1000];
    int type_table[1000];
    sub_root **root;
    point move;
    int total;
    long tmp1,tmp2;
    conection edges[2*r+1];
    double min_length;
    edge now_edge,next_edge;
    conection tmp;
    conection2 cone;
    long link_point,link_flag;
    long tmp_cross[4];
    FILE *fp;
    answer_factor2 answer;
    time_t t;
    srand((unsigned) time(&t));
    clock_t start, end;
    long total_mem=0;
	   start = clock();

    for(int i = 0;i<=1000;i++){
      type_table[i]=-1;
    }
    
    move.init();
    printf("start quad=%d sub_roots=%d k=%d L=%d \n",sizeof(quad_tree),sizeof(conection2),k,L);
    
    if(!input_file){
      init_point(p);
    }else{
      fp = fopen(input_file, "r");
      for(long i=1;i<=Maxpoint;i++){
	fscanf(fp,"%d\n",&p[i].x);
	fscanf(fp,"%d\n",&p[i].y);
      }
      fclose(fp);
    }
    
    if(rand_flag==1){
      move_diff=random((long)L/2);
      for(long i=1;i<=Maxpoint;i++){
	p[i].x = p[i].x + move_diff;
      }  
      move_diff=random((long)L/2);
      for(long i=1;i<=Maxpoint;i++){
	p[i].y = p[i].y + move_diff;
      }
    }

    dissect(p,quad,move,type_table,&total);
    /*
    for(long num=0;num<=total-1;num++){
    printf("quad[%d] total_line_type=%d",num,quad[num].total_line_type);
	    quad[num].print_tree();
	    quad[num].print_lev();
	    printf("quad[%d] total_line(1,1)=%d %d\n",num,quad[num].total_line(1,1),quad[num].portals_num[1]);
	    quad[num].print();
	    printf("============================\n");
    }
    */
       root = new sub_root*[total+1];
      for (long j = 0; j <= total+1; j++ ){
          root[j] = new sub_root[quad[j].hash_table_size()+1];
	  total_mem +=sizeof(sub_root)*quad[j].hash_table_size()+1;
          for(long t=0;t<=quad[j].hash_table_size();t++){
     	  	 	root[j][t].p_cone = new point_conection[quad[j].point_num];
			total_mem +=sizeof(point_conection)* quad[j].point_num;
  	      }
	 
		}
CommentOut(printf("start serach_min_length(quad,total,root,p)\n");)
serach_min_length(quad,total,root,p);
CommentOut(printf("finish serach_min_length(quad,total,root,p)\n");)

start = clock();

answer = search_minimum(quad,root);

/*
tmp2=1;
for(long i=0;i<=quad[tmp2].hash_table_size()-1;i++){
    if((i % 20)==0){
    printf("qq");}
    if(root[tmp2][i].length!=-1.0){
		root[tmp2][i].print2(quad[tmp2].point_num);
  	}
}
tmp2=2;
for(long i=0;i<=quad[tmp2].hash_table_size()-1;i++){
    if((i % 20)==0){
    printf("qq");}
    if(root[tmp2][i].length!=-1.0){
		root[tmp2][i].print2(quad[tmp2].point_num);
  	}
}
tmp2=3;
for(long i=0;i<=quad[tmp2].hash_table_size()-1;i++){
    if((i % 20)==0){
    printf("qq");}
    if(root[tmp2][i].length!=-1.0){
		root[tmp2][i].print2(quad[tmp2].point_num);
  	}
}
tmp2=4;
for(long i=0;i<=quad[tmp2].hash_table_size()-1;i++){
    if((i % 20)==0){
    printf("qq");}
    if(root[tmp2][i].length!=-1.0){
		root[tmp2][i].print2(quad[tmp2].point_num);
  	}
}

*/
//linkを一つつながりにする．
link_point=-1;
for(long j=0;j<=Maxpoint-1;j++){
	if(answer.p_cone[j].link.s_line()!=0){
    	if(link_point==-1){
        	link_point=answer.p_cone[j].c_point();
        }else{
			answer.p_cone[j].link.s_line(0);
			answer.p_cone[j].link.s_number(link_point);
        	link_point=answer.p_cone[j].c_point();
        }

    }else if(answer.p_cone[j].link.e_line()!=0){
    	if(link_point==-1){
        	link_point=answer.p_cone[j].c_point();
        }else{
			answer.p_cone[j].link.e_line(0);
			answer.p_cone[j].link.e_number(link_point);
        	link_point=answer.p_cone[j].c_point();

        }
    }
}
for(long j=0;j<=Maxpoint-1;j++){
	if(answer.p_cone[j].link.s_line()!=0){
		answer.p_cone[j].link.s_line(0);
		answer.p_cone[j].link.s_number(link_point);
    }else if(answer.p_cone[j].link.e_line()!=0){
		answer.p_cone[j].link.e_line(0);
		answer.p_cone[j].link.e_number(link_point);
    }
}
//==================================
Out_Answer_Display(
answer.print(Maxpoint);
)
ans_p_num[0]=0;
ans_p[1]=p[answer.p_cone[0].c_point()];
ans_p_num[1]=answer.p_cone[0].c_point();
next_edge.set(0,answer.p_cone[0].c_point());

for(long i=2;i<=Maxpoint;i++){
  now_edge=next_edge;
  for(long j=0;j<=Maxpoint-1;j++){
    if((answer.p_cone[j].link.st()==now_edge)||(answer.p_cone[j].link.en()==now_edge)){
      link_flag=0;
      for(long t=1;t<=i-1;t++){
	if(ans_p_num[t]==answer.p_cone[j].c_point()){
	  link_flag=1;
	}
      }
      if(link_flag==0){
	ans_p[i]=p[answer.p_cone[j].c_point()];
	ans_p_num[i]=answer.p_cone[j].c_point();
	next_edge.set(0,answer.p_cone[j].c_point());
	break;
      }
    }
  }
}
Out_Answer_Display(
for(long i=1;i<=Maxpoint;i++){
  	printf("Point_num=%d : %d    %d\n",ans_p_num[i],ans_p[i].x,ans_p[i].y);
}

 	printf("Point_num=%d : %d    %d\n",ans_p_num[1],ans_p[1].x,ans_p[1].y);
)
  fp = fopen(output_file, "w");
    for(long i=1;i<=Maxpoint;i++){
	fprintf(fp,"%d %d\n",ans_p[i].x,ans_p[i].y);
	}
	fprintf(fp,"%d  %d\n",ans_p[1].x,ans_p[1].y);

  fclose(fp);

  end = clock();

for(long i=2;i<=Maxpoint;i++){
  ans_length+=distance((double)ans_p[i-1].x,(double)ans_p[i-1].y,(double)ans_p[i].x,(double)ans_p[i].y);
}


    printf("%f end\n",answer.length);
	fprintf(stderr,"%f %d %d %d %f %f %d\n",difftime(end,start),total_mem,no_point_quad,total,ans_length,sucsses/(sucsses+fail),node_num);	
}

int main(int argc,char *argv[]){
  int i;
  char ch_Maxpoint[10],ch_r[5],ch_m[5];
  char *input_file_name=0,*output_file_name=0;
  char *dummy="result.dat";

  for(i=1;i<=argc-1;i++){
    if (strncmp(argv[i], "-p",2) == 0){
      strcpy(ch_Maxpoint,argv[i]+2);
      Maxpoint=atol(ch_Maxpoint);
    }else if (strncmp(argv[i], "-r",2) == 0){ 
      strcpy(ch_r,argv[i]+2);
      r=atol(ch_r);
    }else if (strncmp(argv[i], "-m",2) == 0){
      strcpy(ch_m,argv[i]+2);
      m=atol(ch_m);
    }else if (strncmp(argv[i], "-i",2) == 0){
      delete[] input_file_name;
      input_file_name=new char[strlen(argv[i])];
      strcpy(input_file_name,argv[i]+2);
    }else if (strncmp(argv[i], "-o",2) == 0){   
      delete[] output_file_name;
      output_file_name=new char[strlen(argv[i])];  
      strcpy(output_file_name,argv[i]+2);
    }else if (strncmp(argv[i], "-R",2) == 0){ 
      rand_flag=1;
    }
   
  }
 
  if(!output_file_name){
      delete[] output_file_name;
      output_file_name=new char[strlen(dummy)];  
      strcpy(output_file_name,dummy);
  }

  CommentOut(printf("Maxpoint=%d r=%d m=%d\n",Maxpoint,r,m);)
  //while(1){}
  arora(input_file_name,output_file_name);

}








