

import java.util.*;
import java.io.*;

public class SMFScore{

	//Vector<MusicalNote> score;
	MusicalNote[] score;
	int tracks;
	int division;
	int format;

	int noteNum=0;
	boolean fileUpdata = false;
	final static int noteSize=128;
	final static int channelSize=16;
	public static long totalTime;
	private int[] trackLength;;
	private int longest=-1;
	/*
	public SMFScore(InputStream strm) throws IOException{
		//score = new Vector();
	}
	*/
	
	
	public SMFScore() {
		// TODO 自動生成されたコンストラクター・スタブ
	}
	//パターン読み込み//入力の形式次第で変わる
	//今のところは
	/*
	 * [デルタタイム 高さ]が半角スペースで区切られて複数連なった形をしていると仮定する
	 * はじめの一つだけはデルタタイムがすでに入力されているものとする（
	 */
	public SMFScore(String pat){
		format=0;
		tracks=1;
		division=480;
		int channel=1;
		trackLength=new int[1];
		Vector<MusicalNote> score=new Vector<MusicalNote>();
		StringTokenizer st=new StringTokenizer(pat);

		//高さの情報がない場合そのデルタタイムは破棄する
		try{
			int delta=0;
			int note=Integer.parseInt(st.nextToken());

			score.add(new MusicalNote(channel,0,note));
			while(st.hasMoreTokens()){
				delta+=Integer.parseInt(st.nextToken());
				if(!st.hasMoreTokens())
					break;
				note=Integer.parseInt(st.nextToken());
				score.add(new MusicalNote(channel,delta,note));
			}
			trackLength[0]=delta;
		}catch(NoSuchElementException e){
				
		}
		this.score = (MusicalNote[])score.toArray((new MusicalNote[score.size()]));
	}
	
	
	//可変長表現から値を取り出す
	private static int parseVarLenInt(InputStream strm) throws IOException {
		int oneByte, value = 0; // for storing unsigned byte
		//InputStream.read()--ストリームの終わりに達して読み込むデータがない場合は -1 を返します
		while ( (oneByte = strm.read()) != -1 ) {
			value = value << 7;
			value += oneByte & 0x7f;
			/*
			oneByte ? ? ? ? ? ? ? ?
			0x7f    0 1 1 1 1 1 1 1 
			各bitでANDを取ったものをvalueに入れる
			oneByteの1bit目はステータスビット
			 */
			//SMFのデータ表現でステータスビットがセットされていたら後続バイトがあることを示すので
			//後続バイトが無くなるまで回す。
			if ( (oneByte & 0x80) == 0 )
				break;			
		}
		//ファイルの終わり
		if ( oneByte == -1 )
			return -1;
		return value;
	}
	
	//固定長数値表現から値を取り出す 用途はメタイベントのみ？
	private static int parseVarMetaInt(InputStream strm,int length) throws IOException{
		int val=0;
		for(int i=0;i<length;i++){
			val=val <<8;
			val+=strm.read();			
		}
	
		return val;
	}
	
	public void init(){
		//score=new Vector<MusicalNote>();
		tracks=-1;
		format=-1;
		division=-1;
	}
	
	//ヘッダチャンク内のデータを処理
	public boolean headChunk(InputStream strm) throws IOException{

		byte[] buf ={0,0,0,0};
		strm.read(buf);
		if(buf[0]=='M'&&buf[1]=='T'&&buf[2]=='h'&buf[3]=='d'){
			strm.skip(4);//データサイズをスキップ
			format=strm.read() << 8; //上位byteを入れて左に8bitシフト
			format+=strm.read();     //下位byteを足す

			tracks=strm.read() << 8;
			tracks+=strm.read();

			division=strm.read() << 8;
			division+=strm.read();

		}
		else{
			//System.out.println("Not an SMF file");
			return false;
		}
		noteNum=0;
		return true;
	}

	//トラックチャンク内のデータを処理
	public boolean trackChunk(InputStream strm) throws IOException{
		Vector<MusicalNote> score = new Vector<MusicalNote>();
		Vector[] channelScore = new Vector[channelSize];
		trackLength=new int[tracks];
		for(int i=0;i<channelSize;i++){
			channelScore[i]=new Vector();
		}

		int deltaTotal = 0;
		int deltaTime;
		int velocity;
		int len;
		int tmp=0;
		int tempo=120;
		//ランニングステータスであるフラグ	 true・・ランニングステータス中
		boolean runFlag=false;
		//発音中のノート
		//第一引数がチャネル番号、第二引数がノート番号、内容はScore内の位置
		MusicalNote[][] noteOn=new MusicalNote[channelSize][noteSize];
		int dataSize;
		int high4bits=0,low4bits=0;
		int oneByte;
		byte[] buf={0,0,0,0};
		score.add(new MusicalNote(-1,-1,-1,-1,-1));
		for(int i=0;i<tracks;i++){
			strm.read(buf);
			if(buf[0]=='M'&&buf[1]=='T'&&buf[2]=='r'&buf[3]=='k'){
				dataSize=strm.read() << 8;
				dataSize+=strm.read() << 8;
				dataSize+=strm.read() << 8;
				dataSize+=strm.read() << 8;

			}
			else{
				return false;
			}
			deltaTotal=0;

			//NoteOnノートの初期化
			for(int c=0;c<channelSize;c++){
				for(int n=0;n<noteSize;n++){
					noteOn[c][n]=null;
				}
			}
			while(true){//各処理を抜けたとき次の1byteは必ずデルタタイム
				//デルタタイムの取得
				deltaTime=parseVarLenInt(strm);
				//終了条件
				if(deltaTime==-1){
					break;
				}
				
				deltaTotal+=deltaTime;
				oneByte=strm.read();
				runFlag=true;
				if((oneByte & 0x80)!=0){
					//システムバイト処理
//					上位4bitと下位4bitに分ける
					high4bits=((oneByte & 0xf0) >> 4);
					low4bits=oneByte & 0x0f;
					if(oneByte ==0xf0){
						//システムエクスクルーシブ
						//System.out.println("systemExclusiveMessage");
						while(oneByte!=0xf7){
							oneByte=strm.read();
						}
						continue;
					}
					else if(oneByte==0xf7){
						//エンドオブエクスクルーシブ
						//System.out.println("EOX");
						len=parseVarLenInt(strm);
						//システムリアルタイムメッセージ、ソングポジションポインター、
						//ソングセレクト、MIDIタイムコードを処理する場合はここに記述
						for(int n=0;n<len;n++){
							strm.read();
						}
						continue;
					}
					else if(oneByte==0xff){
						//メタイベント
						//イベントタイプの取得
						oneByte=strm.read();
						if(oneByte==0x2f){
							//End_of_Track
//							16チャネルの音を一つにまとめる
							for(int n=0;n<channelSize;n++){
								if(n==9||channelScore[n].size()==0)//テンポは無視する
									continue;
								trackLength[i]=Math.max(trackLength[i],((MusicalNote)channelScore[n].get(channelScore[n].size()-1)).noteOn);
								score.addAll(score.size(),channelScore[n]);
								score.add(new MusicalNote(-1,-1,-1,-1,-1));
								channelScore[n].clear();
							}

							//イベントサイズを読み飛ばす
							strm.read();
							break;
						}
						else if(oneByte==0x51){//セットテンポ
							int len2=parseVarLenInt(strm);//イベントサイズ3を読み飛ばし
							int tttttt=parseVarMetaInt(strm, len2);
							tempo=(int)(1000000*60.0/tttttt);
							//ttttttは四分音符をマイクロ秒単位で表しているので
							//BPMに変換する。
							//if(deltaTotal==1962892)
							//System.out.println(deltaTotal);
							continue;
						}
						else if(oneByte==0x58){
							strm.read();
							int numerator=strm.read();
							int denomirator=strm.read();
							int cc=strm.read();
							int bb=strm.read();
							//System.out.println("nn : "+numerator);
							//System.out.println("dd : "+denomirator);
							
							continue;

							
						}
						else{
//							イベントサイズの取得
							len=parseVarLenInt(strm);
							for(int j=0;j<len;j++){
								//メタイベントを無視する
								strm.read();
							}
							continue;
						}
					}
					//データバイト取得
					//各処理ごとに取得する事にする(変更に強くするため)
					//oneByte=strm.read();
					runFlag=false;
				}
				/*
//				上位4bitと下位4bitに分ける
				high4bits=((oneByte & 0xf0) >> 4);
				low4bits=oneByte & 0x0f;
				*/
				//MIDIイベント処理
				switch(high4bits){
				case 0x08://noteOff low4bitsはチャネル番号
					//NoteNumber
				/*ランニングステータスの場合すでに上でoneByteは読み込んで
				いるのでここでは読み込まない。以下のステータスの場合も同様
				イベントごとにデータを読み込まずにステータスバイトのとき後続
				１バイトをoneByteに読み込めばrunFlagは使わなくて済む
				*/
					if(!runFlag)
						oneByte=strm.read();
					//Velocity
					strm.read();
					//ノートオフの情報をscoreに付加する
					if(oneByte>noteSize){
						System.out.println(low4bits+" : "+oneByte);
						
					}
					if(noteOn[low4bits][oneByte]!=null){
						noteOn[low4bits][oneByte].setDuration(deltaTotal);
						noteOn[low4bits][oneByte]=null;
					}
					break;
				case 0x09://noteOn low4bitsはチャネル番号
					noteNum++;
//					NoteNumber
					if(!runFlag)
						oneByte=strm.read();
					//Velocity
					velocity=strm.read();
					if(velocity==0x00)
						//noteOfの処理をする
						try{
						if(noteOn[low4bits][oneByte]!=null){
							noteOn[low4bits][oneByte].setDuration(deltaTotal);
							noteOn[low4bits][oneByte]=null;
							break;
						}else{//noteOffとしても使われていない時?
							break;
						}
						}catch(ArrayIndexOutOfBoundsException e){
							throw e;
							//for(int h=0;;)
								//System.out.println(Integer.toHexString(strm.read()));
						}
					//ch9はリズム
					if(low4bits==9){
						continue;
					}
					
					if(noteOn[low4bits][oneByte]!=null){//終了処理されていない音が鳴らされたとき
														//後から鳴った方を無視する(暫定)
						break;
						//無視しない場合durationを付加する
						//noteOn[low4bits][oneByte].setDuration(deltaTotal);
					}

					channelScore[low4bits].add(new MusicalNote(low4bits+1,deltaTotal,oneByte,velocity,tempo));
					noteOn[low4bits][oneByte]=(MusicalNote)channelScore[low4bits].lastElement();
					
					break;
				case 0x0a://ポリフォニックキープレッシャー
					if(!runFlag)
						strm.read();//NoteNumber
					strm.read();//Pressure
					break;
				case 0x0b://コントロールチェンジ
					if(!runFlag)
						strm.read();
					//Control　チャネルモードメッセージを処理するときはここに記述
					strm.read();//Value
					break;
				case 0x0c://プログラムチェンジ
						//ノートオフ、オン処理をする?
					if(!runFlag)
						strm.read();
					break;
				case 0x0d://チャネルプレッシャー
					if(!runFlag)
						strm.read();//Pressure
					break;
				case 0x0e://ピッチベンドチェンジ
					if(!runFlag)
						strm.read();//LSB
					strm.read();//MSB
					break;
				}
			}
			//鳴りっぱなしで終了した時,終了時刻をノートオフの時刻とする
			for(int c=0;c<channelSize;c++){
				for(int j=0;j<noteSize;j++){
					if(noteOn[c][j]!=null){
						noteOn[c][j].setDuration(deltaTotal);
					}
				}
			}
		}
		this.score = (MusicalNote[])score.toArray((new MusicalNote[score.size()]));
		return true;
	}
	//曲中で最後になった音の時間を返す
	public int getLongest(){
		int tmp=-1;
		for(int i=0;i<trackLength.length;i++){
			tmp=Math.max(tmp,trackLength[i]);
		}
		return tmp;
	}
	//トラックiでで最後に鳴った音の時間を返す
	public int getLastTime(int i){
		return trackLength[i];
	}
//	テキストのi番目の音を返す
	MusicalNote noteAt(int i) {
		return score[i];
		//return score.elementAt(i);
	}
	//テキストのサイズを返す
	int size() {
		return score.length;
		//return score.size();
	}
//	分解能を返す
	int getdivision() {
		return division;
	}
	
	public String toString(){
		StringBuffer sb=new StringBuffer();
		sb.append("["+noteNum+"]\n");
		sb.append("tpqn:"+division+"\n");
		sb.append("(chanel,deltaTime,note,duration,velocity,tempo)\n");
		/*
		for(Iterator<MusicalNote> itor=score.iterator();itor.hasNext();){
			sb.append(itor.next().toString()+"\n");
		}
		return sb.toString();
		*/
		for(int i=0;i<score.length;i++){
			sb.append(score[i].toString()+"\n");
		}
		return sb.toString();
	}
	/*
	public void printScore(String str,String filePath) throws IOException{
		BufferedWriter bw = null;
		try {
			if(fileUpdata)//ファイルにデータを追加
				bw=new BufferedWriter(new FileWriter(filePath,true));
			else{//ファイル新規作成
				bw=new BufferedWriter(new FileWriter(filePath,false));
				fileUpdata=true;
			}	
		} catch (FileNotFoundException e) {
			// TODO 自動生成された catch ブロック
			e.printStackTrace();
		} catch (IOException e) {
			// TODO 自動生成された catch ブロック
			e.printStackTrace();
		}
		/*
		bw.write(str+ "["+noteNum+"]");
		bw.newLine();
		bw.write("tpqn:"+division);
		bw.newLine();
		bw.flush();
		bw.write("(chanel,deltaTime,note,duration,velocity,tempo)");
		bw.newLine();
		bw.flush();
		
		for(Iterator itor=score.iterator();itor.hasNext();){
			bw.write(((MusicalNote)itor.next()).toString());
			bw.newLine();
			bw.flush();
		}
		*//*
		bw.write(str+" "+toString());
		bw.newLine();
		bw.close();
		//System.out.println(filePath);
	}
*/

	/*
	//args[0] is readPath args[1] is writePath
	public static void main (String args[]) throws Exception {
		//File file  = new File(args[0]);
		SMFScore score=new SMFScore();
		long time=0;
		//for(int j=0;j<10;j++){
		score.setPath1(args[0]);
		//FileInputStream fistrm= new FileInputStream(args[0]);
		long begin =System.currentTimeMillis();
		int errors=0;
		for(Iterator itor=filePath.iterator();itor.hasNext();){
			try{
				//BufferedInputStream bistrm= new BufferedInputStream(fistrm);
				String file=(String)itor.next();
				BufferedInputStream bistrm=new BufferedInputStream(new FileInputStream(file));
				//System.out.println(file);
				score.init();
				try{
				//long st=System.currentTimeMillis();
				score.headChunk(bistrm);
				score.trackChunk(bistrm);
				//long ms=System.currentTimeMillis();
				//System.out.println("st-ms :"+(ms-st));
				//score.printScore(file,args[1]);
				bistrm.close();
				}catch(ArrayIndexOutOfBoundsException e){
					//System.out.println("error :"+file);
					//errors++;
				}
				
			}
			catch(IOException e){
				//System.out.println(args[0]+" cannot open");
				//throw e;
			}
		}

		long finish=System.currentTimeMillis();
		System.out.println("Total time : "+(finish-begin));
		time+=(finish-begin);
//		System.out.println("error file : "+errors);
		//}
		System.out.println(time/10);
		//score.printScore(args[0]);
		//fistrm= new FileInputStream(args[0]);
		//score.makeHex(fistrm);
		//fistrm.close();
		return;
	}
	*/

}
