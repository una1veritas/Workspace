//
//  SortingAlgorithms.java
//  SortingAlgorithms
//
//  Created by ?? ?? on 07/10/19.
//  Copyright (c) 2007 __MyCompanyName__. All rights reserved.
//
import java.lang.*;
import java.io.*;
import java.util.Random;


class SortingAlgorithms {
	
    private static void maxHeapify(int a[], int i, int end) {
		int larger, tmp;
		
		while ( 2*i + 1 < end ) {
			larger = 2*i + 1;
			if ( (2*i + 2 < end) && (a[2*i + 1] < a[2*i + 2]) ) {
				larger = 2*i + 2;
			}
			if ( a[larger] > a[i] ) {
				tmp = a[larger];
				a[larger] = a[i];
				a[i] = tmp;
				i = larger;
			} else {
				return;
			}
		}
		return;
    }
	
    private static void buildMaxHeap(int a[]) {
		int i;
		
		for (i = (a.length / 2) - 1 ; ! (i < 0); i--) {
			maxHeapify(a, i, a.length);
		}
		return;
    }
	
    public static void heapSort(int a[]) {
		int i;
		int t;
		
		buildMaxHeap(a);
		for (i = a.length - 1; i > 0; i--) {
			// a[0] is always the maximum. 
			t = a[i];
			a[i] = a[0];
			a[0] = t;
			maxHeapify(a, 0, i);
		}
		return;
    }
	
	
	public static void selectionSort(int a[]){
		int i, j, max, t; // i for the length (end+1) of sorted-array.
		
		for(i = a.length - 1; i > 0; i--){
			for(j = 0, max = i; j < i; j++ ){
				if(a[j] > a[max]) {
					max = j;
				}
			}
			t = a[max];
			a[max] = a[i];
			a[i] = t;
			}
		return;
    }
	
/*	
    static void buildMAXHeap(Object array[]) {
		int i;
		
		for (i = (array.length / 2) - 1 ; ! (i < 0); i--) {
			maxHeapify(array, i, array.length);
		}
		return;
    }
	
    private static void maxHeapify(Object array[], int i, int end) {
		int c;
		Object tmp;
		
		while ( 2*i + 1 < end ) {
			c = 2*i + 1;
			if ( 2*i + 2 < end ) {
				if ( ((Comparable) array[2*i + 1]).compareTo(array[2*i + 2]) < 0) {
					c = 2*i + 2;
				}
			}
			if ( ((Comparable) array[c]).compareTo(array[i]) > 0 ) {
				tmp = array[c];
				array[c] = array[i];
				array[i] = tmp;
				i = c;
			} else {
				return;
			}
		}
		return;
    }
	
    public static void heapSort(Object array[]) {
		int i;
		Object tmp;
		
		buildMaxHeap(array);
		for (i = array.length - 1; i > 0; i--) {
			tmp = array[i];
			array[i] = array[0];
			array[0] = tmp;
			maxHeapify(array, 0, i);
		}
		return;
    }
*/	
	
    public static void quickSort(int array[]) {
		quickSort(array, 0, array.length);
    }
    
    public static void quickSort(int array[], int start, int end) {
		int smaller, larger, mid;
		int tmp;
		
		// do nothing if the size of array equals one. 
		if (! (end - start > 1)) {
			return;
		}
		
		// order the first, the middle and the last elements. 
		mid = (end - start) / 2 + start;
		if (array[end - 1] < array[start]) {
			tmp = array[start];
			array[start] = array[end - 1];
			array[end - 1] = tmp;
		}
		if (array[mid] < array[start]) {
			tmp = array[start];
			array[start] = array[mid];
			array[mid] = tmp;
		}
		if (array[end - 1] < array[mid]) {
			tmp = array[mid];
			array[mid] = array[end - 1];
			array[end - 1] = tmp;
		}
		// System.out.println("["+start+", "+mid+", "+(end-1)+"] = "+array[start]+", "+array[mid]+", "+array[end - 1]+". ");
		
		// already enough if the size is no more than three. 
		if (end - start <= 3) {
			return;
		}
		
		// use the middle element as the pivot value. 
		tmp = array[start];
		array[start] = array[mid];
		array[mid] = tmp;
		smaller = start + 1;
		larger = end - 1;
		while ( smaller < larger) {
			if (array[smaller] <= array[start]) {
				smaller++;
			} else {
				// swap array[smaller] with array[larger - 1].
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
		int i, j, tmp;
		
		for(i = 1; i < a.length; i++){
			for(j = a.length - 1; j >= i; j--){
				if(a[j-1] > a[j]) {
					tmp = a[j];
					a[j] = a[j-1];
					a[j-1] = tmp;
				}
			}
		}
		return;
    }
	
	public static void insertionSort(int a[]){
		int i, j, t; // j for the length (end+1) of sorted-array.
		
		for(j = 1; j < a.length; j++) {
			t = a[j];
			for (i = j; i > 0; i--) {
				if (a[i-1] > t) {
					a[i] = a[i-1];
					continue;
				}
			}
			a[i] = t;
		}
		return;
    }
	
	
    public static void recursiveMergeSort(int array[]) {
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
		recursiveMergeSort(bufl);
		recursiveMergeSort(bufr);
		for (i = 0, l = 0, r = 0; i < array.length; i++){
			if ( !(r < bufr.length) ) {
				array[i]=bufl[l];
				l++;
				continue;
			} else if ( !(l < bufl.length) ) {
				array[i]=bufr[r];
				r++;
				continue;
			} else if ( (bufl[l] < bufr[r]) ) {
				array[i]=bufl[l];
				l++;
			} else {
				array[i]=bufr[r];
				r++;
			}
		}
		return;
    }
    
    
    public static void mergeSort(int array[]) {
		int buf[] = new int[array.length];
		int i, l, s, e, cl, cr, ct;
		
		for (l = 1; l < array.length; l = l * 2) {
			for (s = 0; s < array.length; s = s + (2 * l)) {
				e = Math.min(s + (2 * l), array.length);
				for (cl = s, cr = s + l, ct = cl; ct < e; ct++) {
					if (cl < s+l && cr < e) {
						if (array[cl] < array[cr]) {
							buf[ct] = array[cl];
							cl++;
						} else {
							buf[ct] = array[cr];
							cr++;
						}
					} else {
						if (cl < s+l) {
							buf[ct] = array[cl];
							cl++;
						} else {
							buf[ct] = array[cr];
							cr++;
						}
					}
				}
			}
			for (i = 0; i < array.length; i++) 
				array[i] = buf[i];
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
		
		int rep = 1;
		int t, r;
		long worst, best, sum;
		
		if (args.length > 0) {
			size = Integer.parseInt(args[0]);
		}
		if (args.length > 1) {
			range = Integer.parseInt(args[1]);
		}
		
		if (args.length > 2) {
			seed = Integer.parseInt(args[2]);
		}
		
		if (args.length > 3) {
			rep = Integer.parseInt(args[3]);
		}
		
		System.out.println("Size: "+ size +", Range within: 0 -- " + range + ", Initial Seed: " + seed + ", Number of trials: " + rep);
		System.out.println();
		
		array = new int[size];
		
		// Merge Sort
		rnd.setSeed(seed);
		for (worst = 0, best = 0, sum = 0, t = 0; t < rep; t++) {
			r = rnd.nextInt(range) + 2;
			for (int i = 0; i < size; i++) 
				array[i] = rnd.nextInt(r);
			// showing the contents
			if ( t < 1 ) {
				for (int i = 0; i < size; i++) {
					System.out.print(array[i]+", ");
					if (i > 15) {
						System.out.println("... ");
						break;
					}
				}
				System.out.println();
			}
			//
			ela = System.currentTimeMillis();
			SortingAlgorithms.mergeSort(array);
			ela = System.currentTimeMillis() - ela;
			// System.out.println("Merge:   \t"+ (((float)ela)/1000));
			/*
			 for (int ix = 0; ix < array.length; ix++)
			 System.out.print("" + array[ix] + ", ");
			 System.out.println();
			 */
			if (t == 0) {
				worst = ela;
				best = ela;
			} else {
				if (ela > worst) {
					worst = ela;
				}
				if (ela < best) {
					best = ela;
				}
			}
			sum += ela;
		}
		System.out.println("Merge: \tworst " +
						   ((float)worst)/1000 + " sec., \tbest " +
						   ((float)best)/1000 + " sec., \tavr. " +
						   ((int)((float)sum)/rep)/(float)1000 + " sec.");
		
		// Recursive Merge Sort
		/*
		 rnd.setSeed(seed);
		 for (worst = 0, best = 0, sum = 0, t = 0; t < rep; t++) {
			 r = rnd.nextInt(range) + 2;
			 for (int i = 0; i < size; i++) {
				 array[i] = rnd.nextInt(r);
			 }
			 ela = System.currentTimeMillis();
			 SortAlgorithm.recursiveMergeSort(array);
			 ela = System.currentTimeMillis() - ela;
			 // System.out.println("Merge:   \t"+ (((float)ela)/1000));
			 
			 if (t == 0) {
				 worst = ela;
				 best = ela;
			 } else {
				 if (ela > worst) {
					 worst = ela;
				 }
				 if (ela < best) {
					 best = ela;
				 }
			 }
			 sum += ela;
		 }
		 System.out.println("Rmerg: \tworst " +
							((float)worst)/1000 + " m.sec., \tbest " +
							((float)best)/1000 + " m.sec., \tavr. " +
							((int)((float)sum)/rep)/(float)1000 + " m.sec.");
		 */
		
		/*
		 rnd.setSeed(seed);
		 for (int i = 0; i < size; i++) {
			 array[i] = (int) Math.abs(rnd.nextInt()) % range;
		 }
		 ela = System.currentTimeMillis();
		 SortAlgorithm.recursiveMergeSort(array);
		 ela = System.currentTimeMillis() - ela;
		 System.out.println("R-Merge: \t"+ (((float)ela)/1000));
		 */
		
		rnd.setSeed(seed);
		for (worst = 0, best = 0, sum = 0, t = 0; t < rep; t++) {
			r = rnd.nextInt(range) + 2;
			for (int i = 0; i < size; i++) {
				array[i] = rnd.nextInt(r);
				
			}
			ela = System.currentTimeMillis();
			SortingAlgorithms.quickSort(array);
			ela = System.currentTimeMillis() - ela;
			// System.out.println("Quick:  \t"+ (((float)ela)/1000));
			// showing the contents
			/*
			 for (int i = 0; i < size; i++) {
				 System.out.print(array[i]+", ");
				 if (i > 15) {
					 System.out.println("... ");
					 break;
				 }
			 }
			 System.out.println();
			 */
			//
			
			if (t == 0) {
				worst = ela;
				best = ela;
			} else {
				if (ela > worst) {
					worst = ela;
				}
				if (ela < best) {
					best = ela;
				}
			}
			sum += ela;
		}
		System.out.println("Quick: \tworst " +
						   ((float)worst)/1000 + " sec., \tbest " +
						   ((float)best)/1000 + " sec., \tavr. " +
						   ((int)((float)sum)/rep)/(float)1000 + " sec.");
		
		
		rnd.setSeed(seed);
		for (worst = 0, best = 0, sum = 0, t = 0; t < rep; t++) {
			r = rnd.nextInt(range) + 2;
			for (int i = 0; i < size; i++) {
				array[i] = rnd.nextInt(r);
				
			}
			
			ela = System.currentTimeMillis();
			SortingAlgorithms.heapSort(array);
			ela = System.currentTimeMillis() - ela;
			// System.out.println("Heap:   \t"+ (((float)ela)/1000));
			// showing the result
			/*
			 for (int i = 0; i < size; i++) {
				 System.out.print(array[i]+", ");
				 if (i > 15) {
					 System.out.println("... ");
					 break;
				 }
			 }
			 System.out.println();
			 */
			//
			if (t == 0) {
				worst = ela;
				best = ela;
			} else {
				if (ela > worst) {
					worst = ela;
				}
				if (ela < best) {
					best = ela;
				}
			}
			sum += ela;
		}
		System.out.println("Heap: \tworst " +
						   ((float)worst)/1000 + " sec., \tbest " +
						   ((float)best)/1000 + " sec., \tavr. " +
						   ((int)((float)sum)/rep)/(float)1000 + " sec.");
		
		
		
		rnd.setSeed(seed);
		for (worst = 0, best = 0, sum = 0, t = 0; t < rep; t++) {
			r = rnd.nextInt(range) + 2;
			for (int i = 0; i < size; i++) 
				array[i] = rnd.nextInt(r);
			
			ela = System.currentTimeMillis();
			SortingAlgorithms.insertionSort(array);
			ela = System.currentTimeMillis() - ela;
			// System.out.println("Heap:   \t"+ (((float)ela)/1000));
			// showing the result
			//
			if (t == 0) {
				worst = ela;
				best = ela;
			} else {
				if (ela > worst) {
					worst = ela;
				}
				if (ela < best) {
					best = ela;
				}
			}
			sum += ela;
		}
		System.out.println("Insertion: \tworst " +
						   ((float)worst)/1000 + " sec., \tbest " +
						   ((float)best)/1000 + " sec., \tavr. " +
						   ((int)((float)sum)/rep)/(float)1000 + " sec.");
		
		
		rnd.setSeed(seed);
		for (worst = 0, best = 0, sum = 0, t = 0; t < rep; t++) {
			r = rnd.nextInt(range) + 2;
			for (int i = 0; i < size; i++) 
				array[i] = rnd.nextInt(r);
			
			ela = System.currentTimeMillis();
			SortingAlgorithms.selectionSort(array);
			ela = System.currentTimeMillis() - ela;
			// System.out.println("Heap:   \t"+ (((float)ela)/1000));
			// showing the result
			//
			if (t == 0) {
				worst = ela;
				best = ela;
			} else {
				if (ela > worst) 
					worst = ela;
				if (ela < best) 
					best = ela;
			}
			sum += ela;
		}
		System.out.println("Selection: \tworst " +
						   ((float)worst)/1000 + " sec., \tbest " +
						   ((float)best)/1000 + " sec., \tavr. " +
						   ((int)((float)sum)/rep)/(float)1000 + " sec.");
		
		
		//System.out.println((new BufferedReader(new InputStreamReader(System.in))).readLine()+": Ok?");
		
		return;
    }
} 
