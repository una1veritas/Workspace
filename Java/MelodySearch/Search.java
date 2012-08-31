



//SMFScore.java
//SMFScore

//Created by ?? ?? on 06/03/06.
//Copyright (c) 2006 __MyCompanyName__. All rights reserved.

import java.util.*;
import java.io.*;

public class Search {
	int division;

	boolean F=false;
	long[] time={0,0};

	DPTable tbl =null;

	//public static long totalTime;
	private int offset=1;
	private final static int defaultDivision = 480;
	int error=Integer.MAX_VALUE;

	public Search(int err){
		if(err<0)
			error=Integer.MAX_VALUE;
		else
			error=err;
	}

	public void setError(SMFScore score,SMFScore melody){

		int err=-1;
		err=melody.getLastTime(0)*2;

		if(err<defaultDivision*offset)
			error=defaultDivision*offset;

		if((score.getLongest()/2)<melody.getLongest())
			error=Integer.MAX_VALUE;
		else
			error=err;
	}
	public Vector approximateSearchFor(SMFScore score,SMFScore melody, int err ,char type) {
		division=score.getdivision();
		offset=defaultDivision/division;
		tbl = new DPTable(score, melody);
		//setError(score,melody);

		long swatch=0;
		switch(type){
		case 'a':
			swatch=System.currentTimeMillis();
			fillByRowDPTable(tbl, score, melody);
			time[0]+=System.currentTimeMillis()-swatch;
			break;

		case 'b':
			swatch=System.currentTimeMillis();
			quickerFillDPTable2(tbl, score, melody);
			time[0]+=System.currentTimeMillis()-swatch;
			break;

		case 'c':
			swatch=System.currentTimeMillis();
			listed_FillDPTable(tbl, score, melody);
			time[0]+=System.currentTimeMillis()-swatch;
			break;

		case 'd':
			swatch=System.currentTimeMillis();
			listed_quickerFillDPTable(tbl, score, melody);
			time[0]+=System.currentTimeMillis()-swatch;
			break;

		case 'e':
			swatch=System.currentTimeMillis();
			listed_DelLeft_FillDPTable(tbl, score, melody);
			time[0]+=System.currentTimeMillis()-swatch;
			break;

		case 'f':
			swatch=System.currentTimeMillis();
			if(!listed_DelLeftRight_FillDPTable(tbl, score, melody)){
				time[0]+=System.currentTimeMillis()-swatch;
				return new Vector();
			}
			time[0]+=System.currentTimeMillis()-swatch;
			break;

		}

		return traceBackTable(tbl, score, melody, err);		
	}

	public int approximateSearchFor2(SMFScore score,SMFScore melody, int err ) {
		division=score.getdivision();
		offset=defaultDivision/division;
		//offset=1;
		DPTable tbl = new DPTable(score, melody);
		int error=0;
		long swatch=0;

		DPTable tbl2=new DPTable(score,melody);
		listed_FillDPTable(tbl2,score,melody);

		quickerFillDPTable2(tbl,score,melody);
		error+=tbl.equal(tbl2);
		if(error==1){
			System.out.println(tbl.toString(score));
			System.out.println("nnnnnnnnnn");
			System.out.println(tbl2.toString(score));
			System.exit(1);
		}
		tbl = new DPTable(score, melody);
		listed_quickerFillDPTable(tbl,score,melody);
		error+=tbl.equal(tbl2);
		if(error==1){
			System.out.println(tbl.toString(score));
			System.out.println("nnnnnnnnnn");
			System.out.println(tbl2.toString(score));
			System.exit(1);
		}
		tbl = new DPTable(score, melody);
		listed_DelLeft_FillDPTable(tbl,score,melody);
		error+=tbl.equal(tbl2);
		if(error==1){
			System.out.println(tbl.toString(score));
			System.out.println("nnnnnnnnnn");
			System.out.println(tbl2.toString(score));
			System.exit(1);
		}
		tbl = new DPTable(score, melody);
		listed_DelLeftRight_FillDPTable(tbl,score,melody);
		error+=tbl.equal(tbl2);
		if(error==1){
			System.out.println(tbl.toString(score));
			System.out.println("nnnnnnnnnn");
			System.out.println(tbl2.toString(score));
			System.exit(1);

		}

		return error;		
	}

	//|Δ(i,j,k)| リストに関しては関数を呼ばずに直接計算したほうが速いはず
	//チャネル0はスコアを作る際にMIDI上のチャネル+1としているため0になることはないはず
	//チャネルは検索する前に考慮して動かすメソッドを決めている。
	//音の高さはリストごとに分けているため違う音の高さを比較することはなくなっているため。

	int dist(MusicalNote tprev, MusicalNote tcurr, MusicalNote pprev, MusicalNote pcurr) {
		if ( (tprev.channel == 0) || tcurr.channel == 0 ) {
			return Integer.MAX_VALUE;
		}
		//テキストiとi-1の音とパタンjとパタンj-1番目の音が一致しない時
		if ( (tprev.number - tcurr.number) != (pprev.number - pcurr.number) ) {
			return Integer.MAX_VALUE;
		}
		//別の音（ピアノとドラム等）についてのマッチングはしない
		if ( (tprev.channel != tcurr.channel) || pprev.channel != pcurr.channel ){
			return Integer.MAX_VALUE;
		}
		//returns the difference in note duration * defaultDivision
		return (int) Math.abs( (tcurr.noteOn - tprev.noteOn) * offset - (pcurr.noteOn - pprev.noteOn) ) ;
		//return (int) Math.abs( (tcurr.noteOn - tprev.noteOn) - (pcurr.noteOn - pprev.noteOn) ) ;

	}


	class DPTable {
		int d[][];

		DPTable(SMFScore text, SMFScore melody) {
			d = new int[melody.size()][text.size()];
			// tasks for initialization;

			for (int i = 0; i < d.length; i++) {
				Arrays.fill(d[i],0,i,Integer.MAX_VALUE);
			}
			Arrays.fill(d[0], 0, d[0].length, 0);

		}
		public int equal(DPTable t){
			for(int i=1;i<d.length;i++){
				for(int n=0;n<d[0].length;n++){
					if(d[i][n]!=t.d[i][n]){
						System.out.println(i+":"+n);
						return 1;
					}
				}
			}
			return 0;
		}

		public String toString(SMFScore score) {
			StringBuffer buf = new StringBuffer("");
			for (int c = 0; c < d[0].length; c++) {
				buf.append( c + ": " + score.noteAt(c).toString() + " " );
				for (int r = 0; r < d.length; r++) {
					if ( d[r][c] == Integer.MAX_VALUE ) {
						buf.append("\t+++");
					} else {
						buf.append("\t"+d[r][c]);
					}
					buf.append(" ");
				}
				buf.append("\r");
			}
			return buf.toString();
		}
	}


//	List仕様(要素削除 必要な最小距離のみ増やすver)
//	SearchMethodType F
	public boolean listed_DelLeftRight_FillDPTable(DPTable tbl, SMFScore score ,SMFScore melody){
		int row,col,cc;
		int prevNote;
		long Dneg[] = new long[SMFScore.noteSize];
		long Dpos;
		int[] prevLocat = new int[SMFScore.noteSize];
		LinkedList<Integer>[] list=new LinkedList[SMFScore.noteSize];
		boolean isAllMax=true;
		for(int i=0;i<list.length;i++){
			Dneg[i]=Integer.MAX_VALUE;
			list[i]=new LinkedList<Integer>();
		}
		for(col=0;col<score.size();col++){
			if(score.noteAt(col).isEndOfTrack()){
				tbl.d[0][col]=Integer.MAX_VALUE;
			}
			else{
				tbl.d[0][col]=0;
			}
		}

		for(row=1;row<melody.size();row++){
			for(col=0;col<score.size();){
				if(score.noteAt(col).isEndOfTrack()){
					for(int i=0;i<list.length;i++){
						list[i].clear();
					}
					for(cc=col;cc<col+row+1&&cc<score.size();cc++){
						if(score.noteAt(cc).isEndOfTrack()){
							col=cc;
						}
						tbl.d[row][cc]=Integer.MAX_VALUE;
					}
					for(int i=0;i<Dneg.length;i++){
						Dneg[i]=Integer.MAX_VALUE;
						prevLocat[i]=0;
					}
					col=cc;
					continue;
				}

				if(tbl.d[row-1][col-1]!=Integer.MAX_VALUE){
					while(!(list[score.noteAt(col-1).number].isEmpty())){
						int last=list[score.noteAt(col-1).number].getLast();

						long tmp1=(long)tbl.d[row-1][last]+offset*Math.abs(melody.noteAt(row).noteOn-melody.noteAt(row-1).noteOn-(score.noteAt(col).noteOn-score.noteAt(last).noteOn));
						long tmp2=(long)tbl.d[row-1][col-1]+offset*Math.abs(melody.noteAt(row).noteOn-melody.noteAt(row-1).noteOn-(score.noteAt(col).noteOn-score.noteAt(col-1).noteOn));
						if(tmp1 > tmp2){
							list[score.noteAt(col-1).number].removeLast();
						}
						else{
							break;
						}
					}

					list[score.noteAt(col-1).number].add(col-1);
				}
				//一つ前に来るべき音の高さ
				prevNote=score.noteAt(col).number-(melody.noteAt(row).number-melody.noteAt(row-1).number);

				if(prevNote < 0||prevNote>=128){
					tbl.d[row][col]=Integer.MAX_VALUE;

				}	
				else{

					//編集距離を伸ばす
					if(Dneg[prevNote]!=Integer.MAX_VALUE)
						Dneg[prevNote]+=((long)score.noteAt(col).noteOn-(long)score.noteAt(prevLocat[prevNote]).noteOn)*offset;
					prevLocat[prevNote]=col;
					//left-hand side
					while(!list[prevNote].isEmpty()&&
							( melody.noteAt(row).noteOn - melody.noteAt(row-1).noteOn < (score.noteAt(col).noteOn - score.noteAt(list[prevNote].getFirst()).noteOn )*offset)){
						Dneg[prevNote]=Math.min(Dneg[prevNote],(long)tbl.d[row-1][list[prevNote].getFirst()]+dist(score.noteAt(list[prevNote].getFirst()),score.noteAt(col),melody.noteAt(row-1),melody.noteAt(row)));

						list[prevNote].removeFirst();
					}
					//right-hand side
					Dpos=Integer.MAX_VALUE;
					/*
					 for(int i=0;i<list[prevNote].size();i++){
					 int first=list[prevNote].get(i);
					 Dpos=Math.min(Dpos,(long)tbl.d[row-1][first]+dist(score.noteAt(first),score.noteAt(col),melody.noteAt(row-1),melody.noteAt(row)));
					 }
					 */
					if(!list[prevNote].isEmpty()){
						int first=list[prevNote].getFirst();
						Dpos=Math.min(Dpos,(long)tbl.d[row-1][first]+dist(score.noteAt(first),score.noteAt(col),melody.noteAt(row-1),melody.noteAt(row)));
					}
					tbl.d[row][col]=(int) Math.min(Dneg[prevNote],Dpos);

					if(tbl.d[row][col]!=Integer.MAX_VALUE){
						if(tbl.d[row][col]>error)
							tbl.d[row][col]=Integer.MAX_VALUE;
						else
							isAllMax=false;
					}
				}
				col++;
			}
			if(isAllMax){
				if(row==tbl.d.length-1)
					return false;
				row=tbl.d.length-1;
				for(int i=0;i<tbl.d[row].length;i++){
					tbl.d[row][i]=Integer.MAX_VALUE;
				}
				return false;
			}
			isAllMax=true;
		}
		return true;
	}

//	リスト仕様(マイナス側削除)
//	SearchMethodType E
	public boolean listed_DelLeft_FillDPTable(DPTable tbl, SMFScore score,SMFScore melody){
		int row,col,cc;
		int prevNote;
		long[] Dneg=new long[SMFScore.noteSize];
		int[] prevLocat = new int[SMFScore.noteSize];
		long Dpos;
		LinkedList<Integer>[] list=new LinkedList[SMFScore.noteSize];
		boolean isAllMax=true;
		for(int i=0;i<list.length;i++){		
			list[i]=new LinkedList<Integer>();	
		}
		for(col=0;col<score.size();col++){
			if(score.noteAt(col).isEndOfTrack()){
				tbl.d[0][col]=Integer.MAX_VALUE;
			}
			else{
				tbl.d[0][col]=0;
			}
		}

		for(row=1;row<melody.size();row++){
			for(col=0;col<score.size();){
				
				if(score.noteAt(col).isEndOfTrack()){
					for(int i=0;i<list.length;i++){					
						list[i].clear();
					}
					for(cc=col;cc<col+row+1&&cc<score.size();cc++){
						if(score.noteAt(cc).isEndOfTrack()){
							col=cc;
						}
						tbl.d[row][cc]=Integer.MAX_VALUE;
					}

					for(int note=0;note<Dneg.length;note++){
						Dneg[note]=Integer.MAX_VALUE;
						prevLocat[note]=0;
					}

					col=cc;
					continue;
				}

				//マイナス側は再計算するので無限大にする
				Dpos=Integer.MAX_VALUE;

				//一つ前に来るべき音
				prevNote=score.noteAt(col).number-(melody.noteAt(row).number-melody.noteAt(row-1).number);

//				追加する要素のDPテーブル値が無限の場合最小とならないのでリストに入れない
				if(tbl.d[row-1][col-1]!=Integer.MAX_VALUE)
					list[score.noteAt(col-1).number].add(col-1);

				if(prevNote < 0||prevNote>=128){
					tbl.d[row][col]=Integer.MAX_VALUE;		
				}
				else{

//					編集距離を伸ばす
					if(Dneg[prevNote]!=Integer.MAX_VALUE)
						Dneg[prevNote]+=((long)score.noteAt(col).noteOn-(long)score.noteAt(prevLocat[prevNote]).noteOn)*offset;
					prevLocat[prevNote]=col;
					Dpos=Integer.MAX_VALUE;

					while(!list[prevNote].isEmpty()&&
							( melody.noteAt(row).noteOn - melody.noteAt(row-1).noteOn < score.noteAt(col).noteOn - score.noteAt(list[prevNote].getFirst()).noteOn )){

						Dneg[prevNote]=Math.min(Dneg[prevNote],(long)tbl.d[row-1][list[prevNote].getFirst()]+(long)dist(score.noteAt(list[prevNote].getFirst()),score.noteAt(col),melody.noteAt(row-1),melody.noteAt(row)));
						list[prevNote].removeFirst();
					}
					for(Iterator itor=list[prevNote].iterator();itor.hasNext();){
						int pNote=(Integer)itor.next();

						Dpos=Math.min(Dpos,(long)tbl.d[row-1][pNote]+(long)dist(score.noteAt(pNote),score.noteAt(col),melody.noteAt(row-1),melody.noteAt(row)));
					}

					tbl.d[row][col]=(int)Math.min(Dpos,Dneg[prevNote]);
					
					if(tbl.d[row][col]!=Integer.MAX_VALUE){
						if(tbl.d[row][col]>error)
							tbl.d[row][col]=Integer.MAX_VALUE;
						else
							isAllMax=false;
					}
				}
				col++;
			}
			if(isAllMax){

				if(row==tbl.d.length-1)
					return false;
				row=tbl.d.length-1;
				for(int i=0;i<tbl.d[row].length;i++){
					tbl.d[row][i]=Integer.MAX_VALUE;
				}
				return false;
			}
			isAllMax=true;
		}
		return true;
	}

//	リスト仕様(quicker)
//	SearchMethodType D
	public boolean listed_quickerFillDPTable(DPTable tbl, SMFScore score,SMFScore melody){
		int row,col,cc;
		int prevNote;
		int pivot[] =new int[SMFScore.noteSize];
		long[] Dneg=new long[SMFScore.noteSize];
		int[] prevLocat = new int[SMFScore.noteSize];

		long Dpos;
		LinkedList<Integer>[] list=new LinkedList[SMFScore.noteSize];

		boolean isAllMax=true;
		for(int i=0;i<list.length;i++){		
			list[i]=new LinkedList<Integer>();	
		}
		for(col=0;col<score.size();col++){
			if(score.noteAt(col).isEndOfTrack()){
				tbl.d[0][col]=Integer.MAX_VALUE;
			}
			else{
				tbl.d[0][col]=0;
			}
		}

		for(row=1;row<melody.size();row++){
			for(col=0;col<score.size();){
				if(score.noteAt(col).isEndOfTrack()){
					for(int i=0;i<list.length;i++){					
						list[i].clear();
					}
					for(cc=col;cc<col+row+1&&cc<score.size();cc++){
						if(score.noteAt(cc).isEndOfTrack()){
							col=cc;
						}
						tbl.d[row][cc]=Integer.MAX_VALUE;
					}

					for(int note=0;note<Dneg.length;note++){
						pivot[note]=0;
						Dneg[note]=Integer.MAX_VALUE;
						prevLocat[note]=0;
					}

					col=cc;
					continue;
				}

				//マイナス側は再計算するので無限大にする
				Dpos=Integer.MAX_VALUE;

				//一つ前に来るべき音
				prevNote=score.noteAt(col).number-(melody.noteAt(row).number-melody.noteAt(row-1).number);

//				追加する要素のDPテーブル値が無限の場合最小とならないのでリストに入れない
				if(tbl.d[row-1][col-1]!=Integer.MAX_VALUE)
					list[score.noteAt(col-1).number].add(col-1);

				if(prevNote < 0||prevNote>=128){
					tbl.d[row][col]=Integer.MAX_VALUE;

				}
				else{

//					編集距離を伸ばす
					if(Dneg[prevNote]!=Integer.MAX_VALUE)
						Dneg[prevNote]+=((long)score.noteAt(col).noteOn-(long)score.noteAt(prevLocat[prevNote]).noteOn)*offset;
					prevLocat[prevNote]=col;
					Dpos=Integer.MAX_VALUE;

					for(int i=pivot[prevNote];i<list[prevNote].size();i++){
						if ( melody.noteAt(row).noteOn - melody.noteAt(row-1).noteOn < score.noteAt(col).noteOn - score.noteAt(list[prevNote].get(i)).noteOn ) {
							pivot[prevNote]=i+1;
							Dneg[prevNote]=Math.min(Dneg[prevNote],(long)tbl.d[row-1][list[prevNote].get(i)]+(long)dist(score.noteAt(list[prevNote].get(i)),score.noteAt(col),melody.noteAt(row-1),melody.noteAt(row)));
						}
						else{
							Dpos=Math.min(Dpos,(long)tbl.d[row-1][list[prevNote].get(i)]+(long)dist(score.noteAt(list[prevNote].get(i)),score.noteAt(col),melody.noteAt(row-1),melody.noteAt(row)));
						}
					}
					tbl.d[row][col]=(int)Math.min(Dpos,Dneg[prevNote]);
					
					if(tbl.d[row][col]!=Integer.MAX_VALUE){
						if(tbl.d[row][col]>error)
							tbl.d[row][col]=Integer.MAX_VALUE;
						else
							isAllMax=false;
					}
				}
				col++;
			}
			if(isAllMax){

				if(row==tbl.d.length-1)
					return false;
				row=tbl.d.length-1;
				for(int i=0;i<tbl.d[row].length;i++){
					tbl.d[row][i]=Integer.MAX_VALUE;
				}
				return false;
			}
			isAllMax=true;
		}

		return true;
	}

//	リスト仕様
//	SearchMethodType C
	public boolean listed_FillDPTable(DPTable tbl, SMFScore score ,SMFScore melody) {
		int row,col,cc;
		int prevNote;

		int compNum=0;
		int compMax=0;
		long Dpos;
		LinkedList<Integer>[] list=new LinkedList[SMFScore.noteSize];
		boolean isAllMax=true;
		for(int n=0;n<list.length;n++){
			list[n]=new LinkedList<Integer>();

		}
		for(col=0;col<score.size();col++){
			if(score.noteAt(col).isEndOfTrack()){
				tbl.d[0][col]=Integer.MAX_VALUE;
			}
			else{
				tbl.d[0][col]=0;
			}
		}

		for(row=1;row<melody.size();row++){
			for(col=0;col<score.size();){
				compNum=0;
				if(score.noteAt(col).isEndOfTrack()){
					for(int i=0;i<list.length;i++){
						list[i].clear();
					}
					for(cc=col;cc<col+row+1&&cc<score.size();cc++){
						if(score.noteAt(cc).isEndOfTrack()){
							col=cc;
						}
						tbl.d[row][cc]=Integer.MAX_VALUE;
					}
					col=cc;
					continue;
				}

				//一つ前に来るべき音
				prevNote=score.noteAt(col).number-(melody.noteAt(row).number-melody.noteAt(row-1).number);
				//int channel=score.noteAt(col).channel-1;
//				追加する要素のDPテーブル値が無限の場合最小とならないのでリストに入れない
				if(tbl.d[row-1][col-1]!=Integer.MAX_VALUE)
					list[score.noteAt(col-1).number].add(col-1);

				if(prevNote < 0||prevNote>=128){
					tbl.d[row][col]=Integer.MAX_VALUE;
				}

				else{
					Dpos=Integer.MAX_VALUE;
					for(Iterator itor=list[prevNote].iterator();itor.hasNext();){
						compNum++;
						int pNote=(Integer)itor.next();
						long tmp=(long)tbl.d[row-1][pNote]+dist(score.noteAt(pNote),score.noteAt(col),melody.noteAt(row-1),melody.noteAt(row));
						Dpos=Math.min(Dpos,tmp);
					}
					tbl.d[row][col]=(int)Dpos;
					
					if(tbl.d[row][col]!=Integer.MAX_VALUE){
						if(tbl.d[row][col]>error)
							tbl.d[row][col]=Integer.MAX_VALUE;
						else
							isAllMax=false;
					}
				}
				col++;
			}
			if(isAllMax){
				System.out.println(compMax);
				if(row==tbl.d.length-1)
					return false;
				row=tbl.d.length-1;
				for(int i=0;i<tbl.d[row].length;i++){
					tbl.d[row][i]=Integer.MAX_VALUE;
				}
				return false;
			}
			isAllMax=true;
		}
		System.out.println(compMax);
		return true;
	}

//	SearchMethodType B
	public boolean quickerFillDPTable2(DPTable tbl, SMFScore score ,SMFScore melody) {
		int row, col, cc; //, lb;
		int prevNote;
		int compMax=0;
		int compNum=0;
		int pivot[] = new int[128];
		long Dneg[] = new long[128];

		int[] prevLocat = new int[128];
		long Dpos;
		boolean isAllMax=true;
		for (col = 0; col < score.size(); col++) {
			if ( score.noteAt(col).isEndOfTrack() ) {
				tbl.d[0][col] = Integer.MAX_VALUE;
			} else {
				tbl.d[0][col] = 0;
			}
		}

		// for row > 0
		for (row = 1; row < melody.size(); row++) {
			for (col = 0 /*row */; col < score.size(); ) {
				compNum=0;
				if ( score.noteAt(col).isEndOfTrack() || col == 0 ) {
					// the channel has begun, or has been changed; initialize as the first column of the table.
					for (int note = 0; note < Dneg.length; note++) {
						pivot[note] = col+1;
						Dneg[note] = Integer.MAX_VALUE;
						prevLocat[note]=0;
					}

					for (cc = col; cc < col + row + 1 && cc < score.size() ; cc++) {
						if (score.noteAt(cc).isEndOfTrack())
							col = cc;
						tbl.d[row][cc] = Integer.MAX_VALUE;
					}
					col = cc;
					continue;
				}
				Dpos = Integer.MAX_VALUE;

				prevNote = score.noteAt(col).number - ( melody.noteAt(row).number - melody.noteAt(row - 1).number );


				if ( prevNote < 0 || prevNote >= 128 ) {
					tbl.d[row][col] = Integer.MAX_VALUE;
				} else {//pivot[prevNote]は一つ前の高さprevNoteであるノートを調べ始める位置(マイナスからプラスに変わる位置）

					//編集距離を伸ばす
					if(Dneg[prevNote]!=Integer.MAX_VALUE)
						Dneg[prevNote]+=((long)score.noteAt(col).noteOn-(long)score.noteAt(prevLocat[prevNote]).noteOn)*offset;
					prevLocat[prevNote]=col;
					for (cc = pivot[prevNote]; cc < col; cc++) {
						compNum++;
						if ( melody.noteAt(row).noteOn - melody.noteAt(row-1).noteOn < score.noteAt(col).noteOn - score.noteAt(cc).noteOn ) {
							//マイナス側
							//pivotはprevNoteの音のみだった場合に±が変わる場所ではなくて
							//切り替わる場所はprevNote以外の音でも良い？
							pivot[prevNote] = cc + 1;
							//音が違う場合は∞が帰ってくる(dist)
							long tmp = ((long) tbl.d[row-1][cc]) + dist(score.noteAt(cc), score.noteAt(col), melody.noteAt(row-1), melody.noteAt(row));
							if ( tmp < Dneg[prevNote] ) {
								Dneg[prevNote] = tmp;
							}
						} else {
							Dpos = Math.min(Dpos, ((long) tbl.d[row-1][cc]) + dist(score.noteAt(cc), score.noteAt(col), melody.noteAt(row-1), melody.noteAt(row)) );
						}
					}
					tbl.d[row][col] = (int) Math.min( Dneg[prevNote], Dpos );
					compNum++;
					
					if(tbl.d[row][col]!=Integer.MAX_VALUE){
						if(tbl.d[row][col]>error)
							tbl.d[row][col]=Integer.MAX_VALUE;
						else
							isAllMax=false;
					}
				}
				col++;
			}
			if(isAllMax){
				System.out.println(compMax);
				if(row==tbl.d.length-1)
					return false;
				row=tbl.d.length-1;
				for(int i=0;i<tbl.d[row].length;i++){
					tbl.d[row][i]=Integer.MAX_VALUE;
				}
				return false;
			}
			isAllMax=true;
		}
		System.out.println(compMax);
		return true;
	}


//	横にテーブルを作る方法（単純な場合)
//	SearchMethodType a
	public boolean fillByRowDPTable(DPTable tbl, SMFScore score,SMFScore melody) {
		int row, col, lb;//lb:テキストを見る開始位置
		int compMax=0;
		int compNum=0;
		long min;
		boolean isAllMax=false;
		for (row = 1; row < melody.size(); row++) {
			for (col = row, lb = col-1; col < score.size(); col++) {
				compNum=0;
				if ( score.noteAt(col).noteOn < score.noteAt(col-1).noteOn ) {
					//トラックの境目
					// the channel has been changed; initialize as the first column of the table. 
					lb = col; 
				}
				min = Integer.MAX_VALUE;

				for ( int cc = lb; cc < col; cc++) {
					compNum++;
					min = Math.min( min, ((long) tbl.d[row-1][cc]) + dist(score.noteAt(cc), score.noteAt(col), melody.noteAt(row-1), melody.noteAt(row)) );
					//System.out.println(min);
					//System.out.println(dist(score.noteAt(cc), score.noteAt(col), melody.noteAt(row-1), melody.noteAt(row)) );

				}
				tbl.d[row][col] = (int) min; // Math.min( D_minus[r][0], D_plus ); 
				
				if(tbl.d[row][col]!=Integer.MAX_VALUE){
					if(tbl.d[row][col]>error)
						tbl.d[row][col]=Integer.MAX_VALUE;
					else
						isAllMax=false;
				}
			}
			if(isAllMax){
				System.out.println(compMax);
				if(row==tbl.d.length-1)
					return false;
				row=tbl.d.length-1;
				for(int i=0;i<tbl.d[row].length;i++){
					tbl.d[row][i]=Integer.MAX_VALUE;
				}
				return false;
			}
			isAllMax=true;
		}
		System.out.println(compMax);
		return true;

	}
//	縦にテーブルを作る方法
	public void fillByColumnDPTable(DPTable tbl, SMFScore score,SMFScore melody) {
		//		
		int row, col, lb;
		long min;

		//totalTime -= System.currentTimeMillis();
		for (col = 1, lb = 0; col < score.size(); col++) {
			for (row = 1; row < melody.size() && (row <= col); row++) {
				//if (score.noteAt(c).channel != score.noteAt(c-1).channel) {
				if ( score.noteAt(col).isEndOfTrack() ) {
					// the channel has been changed; initialize as the first column of the table. 
					lb = col; 
				}
				min = Integer.MAX_VALUE;
				for ( int cc = lb ; cc < col; cc++) {
					min = Math.min( min, ((long) tbl.d[row-1][cc]) + dist(score.noteAt(cc), score.noteAt(col), melody.noteAt(row-1), melody.noteAt(row)) );
					//System.out.println(min);
				}
				tbl.d[row][col] = (int) min; // Math.min( D_minus[r][0], D_plus ); 
			}
		}
		return; 
	}





	public Vector traceBackTable(DPTable tbl, SMFScore score,SMFScore mel, int err) {
		int col, row;
		int occurrence, cc;
		long best;
		Vector occurrences = new Vector(); // the empty Vector

		if ( err == -1 ) { // find the left-most occurrence with the minimum distance
			occurrence = mel.size() - 1;
			for ( col = 0 + mel.size(); col < score.size(); col++) {
				if ( tbl.d[mel.size() - 1][col] < tbl.d[mel.size() - 1][occurrence] ) {
					occurrence = col;
				}
			}
		} else { // find the left-most occurrence within the specified distance
			occurrence = score.size();
			for ( col = 0 + mel.size(); col < score.size(); col++) {
				if ( tbl.d[mel.size() - 1][col] <= err ) {
					occurrence = col;
					break;
				}
			}
		}

		if ( occurrence == score.size() /* exceeded the size of text */) {
			//occindex[0] = -1;
			//return Integer.MAX_VALUE;
			return occurrences; // returns the empty Vector "occurrences"
		}
		//
		occurrences.addElement(new int[mel.size()]);
		((int[])occurrences.lastElement())[mel.size() - 1] = occurrence;
		//System.out.println(occurrence);
		best = tbl.d[mel.size() - 1][occurrence];
		for ( row = mel.size() - 1, col = occurrence; row > 0; row--) {
			for ( cc = col - 1; ! (cc < 0); cc--) {
				//System.out.print(cc+", ");
				if ( ((long)tbl.d[row-1][cc]) == best - dist(score.noteAt(cc), score.noteAt(col), mel.noteAt(row-1), mel.noteAt(row)) ) {
					break;
				}
			}
			if ( cc < 0 )
				return new Vector();
			//	return new int[0];
			//System.out.println(tbl);
			best -= dist(score.noteAt(cc), score.noteAt(col), mel.noteAt(row-1), mel.noteAt(row));
			col = cc;
			((int[])occurrences.lastElement())[row-1] = col;
		}
		//
		return occurrences; 
	}
	/*
	 static void printTable(DPTable table,int text,int pattern ,boolean flag){
	 int i;
	 StringBuffer sb=new StringBuffer();
	 for(i=0;i<pattern;i++){
	 for(int j=0;j<text;j++){
	 try{
	 if(table.d[i][j]<Integer.MAX_VALUE){
	 System.out.printf("%6d",table.d[i][j]);
	 sb.append("\t"+table.d[i][j]);
	 }
	 else{
	 System.out.print("     m");
	 sb.append("\t"+"m");
	 }
	 }catch(UnknownFormatConversionException e){
	 System.out.println(table.d[i][j]);
	 }
	 if(j==text-1){
	 System.out.println("");
	 sb.append("\n");
	 }

	 }
	 }
	 try {
	 BufferedWriter bw=new BufferedWriter(new FileWriter("d:/test.txt",flag));
	 bw.write(sb.toString());
	 bw.newLine();
	 bw.close();
	 } catch (IOException e) {
	 // TODO 自動生成された catch ブロック
	  e.printStackTrace();
	  }
	  }
	 */
	static void printFullTable(DPTable table){
		int i;
		for(i=0;i<table.d.length;i++){
			for(int j=0;j<table.d[0].length;j++){
				try{
					if(table.d[i][j]<Integer.MAX_VALUE){
						System.out.printf("%6d",table.d[i][j]);
					}
					else{
						System.out.print("     m");
					}
				}catch(UnknownFormatConversionException e){
					System.out.println(table.d[i][j]);
				}
			}
			System.out.println("");
		}

	}

	static void printSubTable(DPTable table,int C){
		int i;

		int end=C+30;
		int start=C-30;
		if(start<0){
			start=0;
		}
		if(end>table.d[0].length){
			end=table.d[0].length;
		}
		System.out.println();
		for(int j=start;j<table.d[0].length&&j<end;j++){
			System.out.printf("%7d",j);
		}
		System.out.println();
		for(i=0;i<table.d.length;i++){
			for(int j=start;j<table.d[0].length&&j<end;j++){
				try{
					if(table.d[i][j]<Integer.MAX_VALUE){
						System.out.printf("%7d",table.d[i][j]);
					}
					else{
						System.out.printf("%7c",'m');
					}
				}catch(UnknownFormatConversionException e){
					System.out.println(table.d[i][j]);
				}
			}
			System.out.println("");
		}

	}

	/*
	 public static void main (String args[]) throws Exception {
	 // insert code here...
	  SMFScore smfscore;
	  int melodyErr;
	  System.out.println("Hello World!");

	  long t2 = 0;
	  // Get the file name list
	   Vector fileNames = new Vector();
	   File file = new File(args[args.length-1]);
	   String files[], fileName;
	   if ( file.isDirectory() ) {
	   files = file.list();
	   for (int i = 0; i < files.length; i++) {
	   if ( (! files[i].startsWith(".")) &&  files[i].endsWith(".mid") ) {
	   fileNames.add(file.getPath() + File.separatorChar + files[i]);
	   }
	   }
	   } else {
	   fileNames.add(file.getPath());
	   }

	   SMFScore melody = new SMFScore(args[0]);
	   melodyErr = Integer.parseInt(args[1]);
	   System.out.println(melody);
	   //int edist;
	    Vector occurrences;
	    int occurrence[];
	    //int occurrence[] = new int[melody.size()];

	     long stopwatch = System.currentTimeMillis();
	     java.text.DecimalFormat fmt = new java.text.DecimalFormat();
	     // perform search
	      //SMFScore.totalTime = 0;
	       for (Iterator i = fileNames.iterator(); i.hasNext(); ) {
	       fileName = (String) i.next();
	       t2 -= System.currentTimeMillis();			
	       FileInputStream istream = new FileInputStream( fileName );
	       try {
	       smfscore = new SMFScore(new BufferedInputStream(istream));
	       } catch (IOException e) {
	       System.out.println(fileName+": ");
	       throw e;
	       }
	       istream.close();
	       t2 += System.currentTimeMillis();
	       //System.out.println("For parsing the score: " + (-stopwatch) );

	        if (smfscore.score.size() == 0) {
	        System.out.println(fileName + ": unknown file format!");
	        System.out.println();
	        continue;
	        } else {
	        System.out.println(fileName + " (" + smfscore.score.size() + ") ");
	        }
	        //
	         occurrences = smfscore.approximateSearchFor(melody, melodyErr );
	         //
	          //SMFScore.totalTime += System.currentTimeMillis();
	           if ( occurrences.isEmpty() ) { 
	           continue;
	           }
	           occurrence = (int[]) occurrences.lastElement();

	           System.out.println();
	           System.out.println(fileName);
	           System.out.println("TPQN = " + smfscore.division + ", the total number of notes = " + smfscore.score.size() /* + ", millisecs to find: " + (-stopwatch) *//* );
	           System.out.print("Ch. "+ smfscore.score.noteAt(occurrence[0]).channel + " from " + occurrence[0] + "th note: ");
	           int k, j;
	           for ( k = 0, j = occurrence[k]; j <= occurrence[occurrence.length - 1]; j++) {
	           if ( smfscore.score.noteAt(j).channel != smfscore.score.noteAt(occurrence[0]).channel )
	           continue;
	           if ( j == occurrence[k] ) {
	           System.out.print("*");
	           k++;
	           }
	           System.out.print(""+smfscore.score.noteAt(j)+", ");
	           }

	           System.out.println();
	           System.out.println();
	           //System.out.println();
	            }
	            //
	             //System.out.println("\r" + "Total time: " + SMFScore.totalTime );
	              System.out.println("Total input time: " + t2 );
		System.out.println("Execution time: " + (System.currentTimeMillis() - stopwatch) );
		//
		return;
	}
	            */
}
