import java.lang.*;
import java.io.*;
import java.util.Random;


class Sort {
    /*
      static int lg(int i) {
      return (int) Math.ceil(Math.log((double) i) / Math.log(2));
      }
    */

    public static void quickSort(int array[]) {
	quickSort(array, 0, array.length);
    }

    public static void quickSort(int array[], int start, int end) {
	int smaller, larger;
	int pivot, tmp;

	// do nothing if the size of array equals one.
	if (! (end - start > 1)) {
	    return;
	}
	// order the first and the last elements.
	if (array[end - 1] < array[start]) {
	    tmp = array[start];
	    array[start] = array[end - 1];
	    array[end - 1] = tmp;
	}
	// already enough if the size of array is two.
	if (end - start == 2) {
	    return;
	}
	// preserve the first element and use it as the pivot value.
	pivot = start;
	smaller = start + 1;
	larger = end - 1;
	while ( smaller < larger) {
	    if (array[smaller] <= array[pivot]) {
		smaller++;
	    } else {
		larger--;
		tmp = array[smaller];
		array[smaller] = array[larger];
		array[larger] = tmp;
	    }
	}
	// divide into two arrays; the latter always has at least one element.
	quickSort(array, start, smaller);
	quickSort(array, smaller, end);
	return;
    }

    public static void bubbleSort(int a[]){
	int i,j,tmp;
	for(i=1;i < a.length;i++){
	    for(j=a.length-1 ; j>=i; j--){
		if(a[j-1]>a[j]){
		    tmp=a[j];
		    a[j]=a[j-1];
		    a[j-1]=tmp;
		}
	    }
	}
	return;
    }

    public static void rMergeSort(int array[]) {
	int i, l, r;
	int bufl[] = new int[array.length/2];
	int bufr[] = new int[array.length/2+(array.length%2)];

	if (array.length <= 1)
	    return;
	for (i = 0; i < array.length/2; i++) {
	    bufl[i] = array[i];
	}
	for (r = 0; i < array.length; i++, r++) {
	    bufr[r] = array[i];
	}
	rMergeSort(bufl);
	rMergeSort(bufr);
	for (i = 0, l = 0, r = 0; i < array.length; i++){
          if ( !(r < bufr.length) ) {
            array[i] = bufl[l];
            l++;
            continue;
          }
          if ( !(l < bufl.length) ) {
            array[i] = bufr[r];
            r++;
            continue;
          }
          if ( bufl[l] < bufr[r] ) {
            array[i] = bufl[l];
            l++;
          } else {
            array[i] = bufr[r];
            r++;
          }
	}
	return;
    }


    public static void mergeSort(int array[]) {
	int buf[] = new int[array.length];
	int i, w, s, e, cl, cr, ct;

	for (w = 1; w < array.length; w = w * 2) {
	    //System.out.println("W: "+w);
	    for (s = 0; s < array.length; s = s + (2 * w)) {
		e = s + (2*w);
				//System.out.println(s+"-"+e);
		cl = s;
		cr = s + w;
		for (ct = s; ct < Math.min(s+(2*w), array.length); ct++) {
		    //System.out.println("cl: "+cl+ "  cr: "+cr+ "  ct: "+ct);
		    if (! (cr < array.length)) {
    			buf[ct] = array[cl];
	    		cl++;
		    	continue;
		    }
		    if (cl < s+w && cr < s+(2*w)) {
    			if (array[cl] < array[cr]) {
	    		    buf[ct] = array[cl];
		    	    cl++;
		    	} else {
		    	    buf[ct] = array[cr];
		    	    cr++;
		    	}
		    } else {
			    if (cl < s+w) {
			        buf[ct] = array[cl];
			        cl++;
			    } else {
			        buf[ct] = array[cr];
			        cr++;
			    }
		    }
		}
	    }
	    for (i = 0; i < array.length; i++) {
		array[i] = buf[i];
	    }
	    //System.out.println();
	}
	return;
    }


    public static void main(String args[]) throws Exception {
	int array[];
	int size = 10;
	Random rnd = new Random();
	long ela;
	int range = 101;
	long seed = System.currentTimeMillis();

	if (args.length > 0) {
	    size = Integer.parseInt(args[0]);
	}
	if (args.length > 1) {
	    range = Integer.parseInt(args[1]);
	}

	if (args.length > 2) {
	    seed = Integer.parseInt(args[2]);
	}

	array = new int[size];

	rnd.setSeed(seed);
	for (int i = 0; i < size; i++) {
	    array[i] = (int) Math.abs(rnd.nextInt()) % range;
	}

	ela = System.currentTimeMillis();
	Sort.quickSort(array);
	ela = System.currentTimeMillis() - ela;
	System.out.println("Quick Sort: "+ (((float)ela)/1000));
	/* showing the result
	for (int i = 0; i < size ; i++) {
	    System.out.print(array[i]+", ");
	}
	System.out.println();
        */


	rnd.setSeed(seed);
	for (int i = 0; i < size; i++) {
	    array[i] = (int) Math.abs(rnd.nextInt()) % range;
	}

	ela = System.currentTimeMillis();
	Sort.mergeSort(array);
	ela = System.currentTimeMillis() - ela;
	System.out.println("Merge Sort: "+ (((float)ela)/1000));


	rnd.setSeed(seed);
	for (int i = 0; i < size; i++) {
	    array[i] = (int) Math.abs(rnd.nextInt()) % range;
	}

	ela = System.currentTimeMillis();
        Sort.rMergeSort(array);
 	ela = System.currentTimeMillis() - ela;
	System.out.println("Recursive Merge Sort: "+ (((float)ela)/1000));

	rnd.setSeed(seed);
	for (int i = 0; i < size; i++) {
	    array[i] = (int) Math.abs(rnd.nextInt()) % range;
	}
	ela = System.currentTimeMillis();
	Sort.bubbleSort(array);
	ela = System.currentTimeMillis() - ela;
	System.out.println("Bubble Sort: "+ (((float)ela)/1000));

	//System.out.println((new BufferedReader(new InputStreamReader(System.in))).readLine()+": Ok?");

	return;
    }
}
