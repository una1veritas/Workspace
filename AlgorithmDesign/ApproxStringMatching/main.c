#include <stdio.h>
#include <string.h>

int rdist(char s[], int m, char t[], int n) {
	int a, b, c;
	if (m == 0 && n == 0)
		return 0;
	if (m == 0)
		return n;
	if (n == 0)
		return m;
	a = rdist(s, m, t, n-1) + 1;
	b = rdist(s, m-1, t, n) + 1;
	c = rdist(s, m-1, t, n-1) + ((s[m-1] == t[n-1])? 0 : 1);
	return (a < b ? (a < c ? a: c): (b < c ? b : c));
}

int dpdist(char s[], int m, char t[], int n) {
	int d[m+1][n+1];
	int i, j, ins, del, repl;
	
	for(i = 0; i < m+1; i++) 
		d[i][0] = i;
	for(j = 0; j < n+1; j++) 
		d[0][j] = j;
	
	for(i = 1; i < m+1; i++) {
		for (j = 1; j < n+1; j++) {
			ins = d[i-1][j]+1;
			del = d[i][j-1]+1;
			repl = d[i-1][j-1] + (s[i-1] == t[j-1] ? 0 : 1);
			d[i][j] = ins < del ? (ins < repl ? ins : repl) : (del < repl ? del : repl);
		}
	}
	// show DP table 
	/*
	for(i = 0; i <= m; i++) {
		for (j = 0; j <= n; j++) { 
			printf("%d\t", d[i][j]);
		}
		printf("\n");
	}
	 */
	return d[m][n];
}

int main (int argc, const char * argv[]) {
	char * s = (char *) argv[1], * t = (char *) argv[2];
	int m = strlen(s), n = strlen(t);
	int d;
	
	printf("Input: %s (length %d), %s (length %d)\n", s, m, t, n);
	
	d = rdist(s, m, t, n);
	printf("Edit distance (by recursion): %d\n", d);
	
	d = dpdist(s, m, t, n);
	printf("Edit distance (by DP): %d\n", d);
	
    return 0;
}
