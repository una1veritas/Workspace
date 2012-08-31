

import java.awt.List;
import java.io.*;
import java.util.*;

//楽譜をトラックごとにチャネル別にソートした場合のコントローラ
public class MusicSearchControler implements Runnable{
	private Vector<String> filePath=null;
	static int fileSearchDepth=5;
	static String writePath=null;
	private static int which=0;
	SMFScore score;
	SMFScore melody;
	List list;
	private boolean restart=false;
	public MusicSearchControler(Vector<String> path,String mel){
		melody=new SMFScore(mel);
		filePath=path;
	}
	public MusicSearchControler() {
		// TODO 自動生成されたコンストラクター・スタブ
	}
	public void restart(){
		restart=true;
	}
	
	private static Vector<String>[] setPath1(String path,int Thread_num){
		Vector<String>[] filePath=new Vector[Thread_num];
		for(int i=0;i<Thread_num;i++){
			filePath[i]=new Vector<String>();
		}
		setPath(path,fileSearchDepth,filePath);
		return filePath;
	}
	private static void setPath(String path,int depth,Vector[] filePath){
		if(depth==-1)
			return;
		File file = new File(path);
		String[] fileNames=file.list();
		if(file.isFile()){
			filePath[which%filePath.length].add(file.getAbsolutePath());
			which++;
			return;
		}
		if(fileNames==null){
			System.out.println(file);
			return;
		}
		for(int i=0;i<fileNames.length;i++){
			file=new File(path,fileNames[i]);
			if(file.isDirectory()){
				setPath(file.getAbsolutePath(),depth-1,filePath);
			}
			else{
				filePath[which%filePath.length].add(file.getAbsolutePath());
				which++;
			}
		}
	}
	public static void write(String st,boolean flag,String path){
		try {
			BufferedWriter bw=new BufferedWriter(new FileWriter(path,flag));
			bw.write(st);
			bw.newLine();
			bw.flush();
			bw.close();
		} catch (IOException e) {
			// TODO 自動生成された catch ブロック
			e.printStackTrace();
		}
	}

	private static int err=0;
	private static char type='x';
	private static boolean notInit=true;
	long noteNum=0;
	int hit=0;

	
	public static void init(String error,char typ,int rad){
		//melody=new SMFScore(mel);
		err=Integer.parseInt(error);
		type=typ;
		notInit=false;
	}
	
	public void set(String args[],List list){
		score=new SMFScore();
		melody=new SMFScore(args[1]);
		err=Integer.parseInt(args[2]);
		type='f';
		//for(int j=0;j<10;j++){
		filePath=setPath1(args[0],1)[0];
		this.list=list;
		restart=false;
		notInit=false;
	}
	
	public void run(){
		if(notInit)
			return;
		StringBuffer result=new StringBuffer();
		score=new SMFScore();
		hit=0;
		Vector occurrences=null;
		BufferedInputStream bistrm=null;;
		Search search = new Search(err);
		int count=0;
		for(Iterator<String> itor=filePath.iterator();itor.hasNext();){
			if(restart)
				return;
			String file=itor.next();
			try{
				bistrm=new BufferedInputStream(new FileInputStream(file));
			
				score.init();
				if(!score.headChunk(bistrm)){
					bistrm.close();
					continue;
				}
				if(!score.trackChunk(bistrm)){
					bistrm.close();
					continue;
				}
				bistrm.close();
				if(score.size()<
						melody.size()){
					continue;
				}
				count++;
	
			noteNum+=score.size();
			occurrences=search.approximateSearchFor(score,melody,err,type);
			
			if(occurrences==null||occurrences.isEmpty())
				continue;
			
			result.append(file + " : ");
			int[] occurrence = (int[]) occurrences.lastElement();
			int k, j;
			list.add(file+" : "+score.noteAt(occurrence[0]).noteOn);
			for ( k = 0, j = occurrence[k]; j <= occurrence[occurrence.length - 1]; j++) {
				if ( score.noteAt(j).channel != score.noteAt(occurrence[0]).channel )
					continue;
				if ( j == occurrence[k] ) {
					result.append("*");
					k++;
				}
				result.append(""+score.noteAt(j)+", ");
				if(j==occurrence[occurrence.length-1]){
					result.append(" : "+search.tbl.d[search.tbl.d.length-1][j]+"\n");
				
				}
			}
			
			hit++;
			}catch(ArrayIndexOutOfBoundsException e){
			}catch(NoSuchElementException e){
				System.out.println("NoSuchElement");
				continue;
			}catch(IOException e){
				System.out.println("IOException");
			}catch(Exception e){
				System.out.println("exception");
				System.out.println(file);
				e.printStackTrace();
				continue;
			}
		}
		list.add("total : "+count);
		list.add("hit : "+(hit));
		//write(result.toString(),false,"C:\\Documents and Settings\\kizaki\\デスクトップ\\search2\\file\\test.txt");
	}
	public long noteCount(String path){
		filePath=(setPath1(path,1))[0];
		SMFScore score;
		String file=null;
		BufferedInputStream bistrm=null;
		long noteNum=0;
		for(Iterator<String> itor=filePath.iterator();itor.hasNext();){
			
			file=itor.next();
			try{
			bistrm=new BufferedInputStream(new FileInputStream(file));
			score =new SMFScore();
			score.init();
			if(!score.headChunk(bistrm))
				bistrm.close();
			score.trackChunk(bistrm);
			bistrm.close();
			noteNum+=score.noteNum;
			}catch(Exception e){
				continue;
			}
		}
		return noteNum;
	}

	public void MusicSearch(String path,String pat,String error,char type){
		long time;
		filePath=(setPath1(path,1))[0];
		melody=new SMFScore(pat);
		int err=Integer.parseInt(error);
		hit=0;
		Search search=new Search(err);
		time=0;
		long begin =System.currentTimeMillis();
		int errors=0;
		Vector occurrences = null;
		score=new SMFScore();
		int num=0;
		for(Iterator<String> itor=filePath.iterator();itor.hasNext();){
			try{
				String file=itor.next();
				BufferedInputStream bistrm=new BufferedInputStream(new FileInputStream(file));
				score.init();

				try{
					long tmp=System.currentTimeMillis();
					if(!score.headChunk(bistrm)){
						bistrm.close();
						continue;
					}
					if(!score.trackChunk(bistrm)){
						bistrm.close();
						continue;
					}
					bistrm.close();
					if(score.size()<melody.size()){
						continue;
					}
					time+=System.currentTimeMillis()-tmp;

					noteNum+=score.size();
					occurrences=search.approximateSearchFor(score,melody,err,type);
					
					num++;
					if(occurrences.isEmpty())
						continue;
					
					
					System.out.print(file + " : ");
					int[] occurrence = (int[]) occurrences.lastElement();
					int k, j;
					for ( k = 0, j = occurrence[k]; j <= occurrence[occurrence.length - 1]; j++) {
						if ( score.noteAt(j).channel != score.noteAt(occurrence[0]).channel )
							continue;
						if ( j == occurrence[k] ) {
							System.out.print("*");
							k++;
						}
					//	System.out.print(""+score.noteAt(j)+", ");
						if(j==occurrence[occurrence.length-1]){
							System.out.println();
							System.out.println(" : "+search.tbl.d[search.tbl.d.length-1][j]);
						
						}
					}
					
					hit++;
				}catch(ArrayIndexOutOfBoundsException e){
					//System.out.println("error :"+file);
					//errors++;
					//e.printStackTrace();
				}catch(NoSuchElementException e){
					//System.out.println("error : "+file);
					errors++;
					return;
				}
				
				
			}
			catch(IOException e){
				//System.out.println(args[0]+" cannot open");
				//throw e;
			}
			
		}
		
		long finish=System.currentTimeMillis();
		System.out.println((finish-begin)+","+time+","+search.time[0]);
		/*
		System.out.println("Total time : "+(finish-begin));

		System.out.println("errors      : "+errors);
		System.out.println("noteNum     : "+noteNum);
		System.out.println("file total  : "+num);
		System.out.println("hit         : "+hit);
		System.out.println("readingTime : "+time);
		
		System.out.println("nomo        : "+search.mono);
		System.out.println("multi       : "+search.multi);
		System.out.println("      SearchingTime      "); 
		
		for(int i=0;i<type.length();i++){
			switch(type.charAt(i)){
			case 'a':System.out.println("naiveSearch : "+search.time[i]+"ms");
			break;
			case 'b':System.out.println("quickerSearch16 : "+search.time[i]+"ms");
			break;
			case 'c':System.out.println("listedSearch16 : "+search.time[i]+"ms");
			break;
			case 'd':System.out.println("listedQuickerSearch16 : "+search.time[i]+"ms");
			break;
			case 'e':System.out.println("listedDelLeftSearch16 : "+search.time[i]+"ms");
			break;
			case 'f':System.out.println("listedDelLeft&RightSearch16 : "+search.time[i]+"ms");
			break;
			case 'g':System.out.println("listedDelAddSearch : "+search.time[i]+"ms");
			break;
			
			case 'A':System.out.println("quickerSearch : "+search.time[i]+"ms");
			break;
			case 'B':System.out.println("quickerSearchS : "+search.time[i]+"ms");
			break;
			case 'C':System.out.println("listedSearch : "+search.time[i]+"ms");
			break;
			case 'D':System.out.println("listedQuickerSearch : "+search.time[i]+"ms");
			break;
			case 'E':System.out.println("listedDelLeftSearch : "+search.time[i]+"ms");
			break;
			case 'F':System.out.println("listedDelLeftRightSearch : "+search.time[i]+"ms");
			break;
			case 'G':System.out.println("listedDelAddSearch : "+search.time[i]+"ms");
			break;
			}
		}
		
		//score.printScore(args[0]);
		//fistrm= new FileInputStream(args[0]);
		//score.makeHex(fistrm);
		//fistrm.close();
		return;
		*/
	}
}
