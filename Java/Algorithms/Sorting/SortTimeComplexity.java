import java.lang.*;
import java.io.*;
import java.util.Random;


class SortTimeComplexity {

    public static void main(String args[]) 
	throws Exception {

	int array[];
	int size = 100;
	int stopsize = 1000;
	float increment = (float) 2.0;
	Random rnd = new Random();
	long ela;
	long seed = 0;

	int range = 101;
	
	if (args.length > 0) {
	    size = Integer.parseInt(args[0]);
	}
	
	if (args.length > 1) {
	    stopsize = Integer.parseInt(args[1]);
	}
	
	if (args.length > 2) {
	    seed = Integer.parseInt(args[2]);
	}


	System.out.println("size\tmerge\tr-merge\theap\tquick\tselection\tbubble");

	while (! (size > stopsize) ) {
	    for (int cnt = 0; cnt < 100 ; cnt++) {

		array = new int[size];
		if (seed == 0) {
		    seed = System.currentTimeMillis();
		}
		range = (cnt+1) * 4;

		System.out.print(size);
		
		rnd.setSeed(seed);
		for (int i = 0; i < size; i++) {
		    array[i] = Math.abs(rnd.nextInt(range));
		}
		// showing the contents 
		/*
		  for (int i = 0; i < size; i++) {
		  System.err.print(array[i]+", ");
		  if (i > 15) {
		  System.err.println("... ");
		  break;
		  }
		  }
		  System.err.println();
		*/
		//
		ela = System.currentTimeMillis();
		SortAlgorithm.mergeSort(array);
		ela = System.currentTimeMillis() - ela;
		System.out.print("\t" + (((float)ela)/1000));
		
		rnd.setSeed(seed);
		for (int i = 0; i < size; i++) {
		    array[i] = (int) Math.abs(rnd.nextInt()) % range;
		}
		ela = System.currentTimeMillis();
		SortAlgorithm.recursiveMergeSort(array);
		ela = System.currentTimeMillis() - ela;
		System.out.print("\t"+ (((float)ela)/1000));
		
		rnd.setSeed(seed);
		for (int i = 0; i < size; i++) {
		    array[i] = (int) Math.abs(rnd.nextInt()) % range;
		}
		ela = System.currentTimeMillis();
		SortAlgorithm.heapSort(array);
		ela = System.currentTimeMillis() - ela;
		System.out.print("\t"+ (((float)ela)/1000));
		
		rnd.setSeed(seed);
		for (int i = 0; i < size; i++) {
		    array[i] = (int) Math.abs(rnd.nextInt()) % range;
		}
		ela = System.currentTimeMillis();
		SortAlgorithm.quickSort(array);
		ela = System.currentTimeMillis() - ela;
		System.out.print("\t"+ (((float)ela)/1000));
		
		rnd.setSeed(seed);
		for (int i = 0; i < size; i++) {
		    array[i] = (int) Math.abs(rnd.nextInt()) % range;
		}	
		ela = System.currentTimeMillis();
		SortAlgorithm.selectionSort(array);
		ela = System.currentTimeMillis() - ela;
		System.out.print("\t"+ (((float)ela)/1000));
		
		rnd.setSeed(seed);
		for (int i = 0; i < size; i++) {
		    array[i] = (int) Math.abs(rnd.nextInt()) % range;
		}	
		ela = System.currentTimeMillis();
		SortAlgorithm.bubbleSort(array);
		ela = System.currentTimeMillis() - ela;
		System.out.print("\t"+ (((float)ela)/1000));

		System.out.println();
	    }
	    size = (int) (size * increment + 1);
	}
	return;
    }
} 
