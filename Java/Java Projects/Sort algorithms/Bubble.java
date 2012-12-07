class BubbleSort {
	long array[];
	
	public BubbleSort(long size) {
		array = new long[size+1];
		
	}
 sort(int *a ,int n){
  long i,j,tmp;
  for(i=2;i<=n;i++){
    for(j=n;j>=i;j--){
      if(a[j-1]>a[j]){
	tmp=a[j];
	a[j]=a[j-1];
	a[j-1]=tmp;
      }
 //     printf("%d %d \n",i,j);
    }
  }
}

int main(void){
  long n=1000000,i;
  long a[n+1];
  time_t t;
  clock_t start,end;
  srand((unsigned) time(&t));
  
  for(n=10;n<=1000000;n*=10){
  
  for(i=1;i<=n;i++){
    a[i]=random(n*100);
  }
  
  start = clock();
  bublesort(a,n);
  end = clock();
 
  /* for(i=1;i<=n;i++){
    printf("a[%d]=%d\n",i,a[i]);
  }
  */ 
  printf("element = %d time=%f\n",n,difftime(end,start)/CLOCKS_PER_SEC);
  }
}

}