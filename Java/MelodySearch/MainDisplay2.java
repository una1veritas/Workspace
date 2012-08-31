import java.applet.Applet;
import java.util.*;
import java.awt.*;
import java.awt.event.*;
import javax.sound.midi.*;


//インターフェイスメイン
//試作型
//main10に加えてメトロノーム部分の変更
//九州工業大学情報工学部 知能情報工学科 03231028 木崎 貴久
/*
	<applet code="MainDisplay2" width = 820 height = 700>
	</applet>
*/
public class MainDisplay2 extends Applet implements ActionListener,Runnable,MouseListener,AdjustmentListener{
	LinkedList<Integer> llm = new LinkedList<Integer>();//音高情報を保持したLinkedList
	LinkedList<Long> llr=new LinkedList<Long>(),llt=new LinkedList<Long>(),lltc=new LinkedList<Long>();//llr-音長llt-tick情報lltc-tickのコピー
	ArrayList<Integer> xpos = new ArrayList<Integer>(), ypos = new ArrayList<Integer>();//xpos,ypos=確認画面のnoteのx座標のリストnote移動に必要
	Synthesizer synth = null;//シンセサイザーの作成
	MidiChannel chan[] = null;//midiチャネルの作成
	Track track = null,track2 = null;//trackの作成 trackはメロディtrack track2はリズムtrack
	Sequencer seqer = null, seqer2=null;//シーケンサの作成
	Sequence seq = null, seq2 = null;//seqメロディのシーケンス seq2リズムのシーケンス
	Image img,img2;//img2はメトロノーム部分  imgはそれ以外(鍵盤確認画面)のイメージ
	Graphics wx,wx2;//Graphicsの作成 wxは鍵盤 情報出力部分 wx2はメトロノーム部分
	String tempost;//テンポのString表現
	TextField diff2;//検索に対する総tickのずれ
	int note, ir = 0, state = 0, slc=0;//note=音高 ir=リズム入力時の音高の現在位置 state=状態メロディ１リズム２補正３ slc=確認画面上の選択notenumber
	double divx = 1;//確認画面の分割値
	long allstart, start, stop, time, ttime, mtime= 0;//time=リズムの入力時間 ttime=ティック mtime=メロディタイム500(8分音符)刻み
	boolean ncon = false;//ncon=trueならllmに要素が入っている
	Button play,rtm,bs,re,crt,sch,play2;//play=再生停止 rtm=リズム入力 bs=noteBS re=reset crt=correct sch=search
	java.awt.List correct, hitlist; //補正範囲を決定するlist
	Scrollbar temposcr, tickscr; //tempoを設定するスクロールバー 
	Thread th, thread1;//thメトロノーム機能の為のスレッド thread1
	Label expla,tempola,tickla; //ラベル(操作説明) tempotf//tempoの現在値
	String  m[] = {"鍵盤→searchでmelody検索 rhythm又はrhythm→correct →searchでrhythm検索",
				"鍵盤はmelody(音高)の入力のみで音長は入らない。rhythm(音長)を入力するにはrhythmbuttonへ",
				"情報表示画面の縦線は4分音符刻み。最初のrhythmを入力してから最後のrhythmを入力するまでリアルタイムで進む",
				"melody入力時のみ最後尾の音を1音だけ消す",
				"現段階の情報を破棄して1段階前の状態へ遷移 melody→初期化 rhythm→melody correct→rhythm",
				"Listで選択(4or8or16)したによって補正最小音長を決定.それぞれの補正音長に1番近い音のみを補正",
				"現在は入力した情報をTextFieldへ表示"}, playdate;
	//キー情報１．音階２．位置３．黒鍵白鍵
	String t[][] = {{"48","000","0"},{"49","015","1"},{"50","030","0"},{"51","045","1"},{"52","060","0"},
					{"53","090","0"},{"54","105","1"},{"55","120","0"},{"56","135","1"},{"57","150","0"},
					{"58","165","1"},{"59","180","0"},{"60","210","0"},{"61","225","1"},{"62","240","0"},
					{"63","255","1"},{"64","270","0"},{"65","300","0"},{"66","315","1"},{"67","330","0"},
					{"68","345","1"},{"69","360","0"},{"70","375","1"},{"71","390","0"},{"72","420","0"},
					{"73","435","1"},{"74","450","0"},{"75","465","1"},{"76","480","0"},{"77","510","0"},
					{"78","525","1"},{"79","540","0"},{"80","555","1"},{"81","570","0"},{"82","585","1"},
					{"83","600","0"},{"84","630","0"}};
	Long playtick;//再生場所のtick
	MusicSearchControler msc;
	boolean isRunning=false;
	public void start(String[] args){
		if(thread1!=null&&thread1.isAlive()){
			msc.restart();
			try{
				thread1.join();
				hitlist.removeAll();
			}catch(Exception e){
				System.out.println("join fail");
			}
		}
		msc=new MusicSearchControler();
		try {
			msc.set(args,hitlist);
			thread1=new Thread(msc);
			thread1.start();
		} catch (Exception e) {
			System.out.println("thread error");
			e.printStackTrace();
		}
	}
	public void init(){
		try{
			synth = MidiSystem.getSynthesizer();// シンセサイザーを取得
			chan = synth.getChannels();// MIDIチャンネルリストを取得
			seq = new Sequence(Sequence.PPQ,1000);//PPQは四分音符あたりのtick メロディシーケンスの作成
			track = seq.createTrack();//空のトラック作成にはコンストラクタではなく右の式を実行
			seqer = MidiSystem.getSequencer();//デフォルトのシーケンサを取得
			seqer2= MidiSystem.getSequencer();
			seqer.open();//シーケンサを開く
			seqer2.open();
			synth.open();// シンセサイザーを開く
		}catch  (Exception e) {
			e.printStackTrace();
		}
		img  = createImage(680,400);
		wx  = img.getGraphics();
		img2  = createImage(101,20); 
		wx2  = img2.getGraphics();
		setLayout(null);//コンポーネントの配置
		
		add(expla = new Label());//説明Labelの追加
		set(expla,5,2,810,18);
		expla.setText(m[0]);
		
		add(rtm=new Button("rhythm"));//リズムボタンの追加
		set(rtm,700,30,100,100);
		rtm.addMouseListener(this);
		
		add(temposcr = new Scrollbar(Scrollbar.HORIZONTAL,60,10,20,210));//テンポスクロールバー範囲メトロノーム20~200まで
		set(temposcr,700,150,100,15);
		temposcr.addAdjustmentListener(this);
		
		add(tempola=new Label());//テンポ表示ラベルの追加
		set(tempola,700,165,40,20);
		tempola.setText(tempost.valueOf(temposcr.getValue()));
		
		add(correct=new java.awt.List(1,false));//補正範囲ラベルの追加
		correct.add("4分");correct.add("8分");correct.add("16分");
		correct.select(0);
		correct.makeVisible(0);
		set(correct,750,165,50,20);
		
		add(crt=new Button("correct"));//correctの追加
		set(crt,700,185,100,25);
		crt.addActionListener(this);
		
		add(bs=new Button("note backspace"));//noteBSの追加
		set(bs,700,215,100,25);
		bs.addActionListener(this);
		
		add(re=new Button("melody reset"));//resetの追加
		set(re,700,240,100,25);
		re.addActionListener(this);
		
		add(play=new Button("play/stop"));//再生/停止ボタンの追加
		set(play,700,280,100,25);
		play.addActionListener(this);
		
		add(sch=new Button("search"));//検索ボタンの追加
		set(sch,700,310,100,25);
		sch.addActionListener(this);
		
		add(play2=new Button("search play"));
		set(play2,620,500,200,80);
		play2.addActionListener(this);
		
		add(hitlist=new java.awt.List(10,false));
		set(hitlist,100,350,500,300);
		hitlist.addActionListener(this);
		
		add(diff2 = new TextField("0",60));
		set(diff2,620,480,100,20);
		
		add(tickscr = new Scrollbar(Scrollbar.HORIZONTAL,0,10,0,1010));
		set(tickscr,620,620,200,15);
		tickscr.addAdjustmentListener(this);
		
		add(tickla=new Label());
		set(tickla,620,600,100,20);
		tickla.setText(new String().valueOf(tickscr.getValue()));
		
		addMouseListener(this);//アプレットのクリックに対するイベントの追加
		disp();
		th = new Thread(this);//スレッドの使用
		th.start();//スレッドの開始
	}
	public void run(){//スレッドを使用してメトロノーム機能をつける。点滅の間隔は８分音符基準
		try{
			while(true){
				for(int x=50;x<100;x=x+25){
					wx2.setColor(Color.white);
					wx2.fillRect(0,0,101,20);
					wx2.setColor(Color.black);
					wx2.drawLine(50,0,50,20);//メトロノーム基準線
					wx2.setColor(Color.blue);
					wx2.drawLine(x,0,x,20);
					repaint();
					Thread.sleep(60000/temposcr.getValue()/8);//テンポをミリ秒へ変換
				}
				for(int x=100;x>50;x=x-25){
					wx2.setColor(Color.white);
					wx2.fillRect(0,0,101,20);
					wx2.setColor(Color.black);
					wx2.drawLine(50,0,50,20);//メトロノーム基準線
					wx2.setColor(Color.blue);
					wx2.drawLine(x,0,x,20);
					repaint();
					Thread.sleep(60000/temposcr.getValue()/8); 
				}
				for(int x=50;x>0;x=x-25){
					wx2.setColor(Color.white);
					wx2.fillRect(0,0,101,20);
					wx2.setColor(Color.black);
					wx2.drawLine(50,0,50,20);//メトロノーム基準線
					wx2.setColor(Color.blue);
					wx2.drawLine(x,0,x,20);
					repaint();
					Thread.sleep(60000/temposcr.getValue()/8); 
				}
				for(int x=0;x<50;x=x+25){
					wx2.setColor(Color.white);
					wx2.fillRect(0,0,101,20);
					wx2.setColor(Color.black);
					wx2.drawLine(50,0,50,20);//メトロノーム基準線
					wx2.setColor(Color.blue);
					wx2.drawLine(x,0,x,20);
					repaint();
					Thread.sleep(60000/temposcr.getValue()/8); 
				}
			}
		}catch(Exception e){
		}
	}
	public void update(Graphics g){//ちらつきをおこさないようにするupdateメソッドのオーバーライド
		paint(g);
	}
	public void disp(){//画面を構成する関数
		setBackground(Color.lightGray);
		wx.setColor(Color.lightGray);
		wx.fillRect(0,0,800,400);//全面をlightGrayで塗りつぶす
		for (int i=0;i<t.length;i++) {
			if (t[i][2]=="0") { //白鍵
				wx.setColor(Color.white); 
				wx.fillRect(Integer.parseInt(t[i][1]),0,28,100);//白鍵を塗りつぶす
			}
		}
		for (int i=0;i<t.length;i++) {
			if (t[i][2]=="1") {//黒鍵
				wx.setColor(Color.black); 
				wx.fillRect(Integer.parseInt(t[i][1]),0,28,50);
			}
		}
		//ここから確認画面を作成
		wx.setColor(Color.white);
		wx.fillRect(0,120,660,184); //枠の作成横660縦184
		wx.setColor(Color.black);
		//横線
		wx.drawLine(0,180,660,180);
		wx.drawLine(0,240,660,240); //中央線のCの基準線
		wx.setColor(Color.red);
		if(state==0&&ncon==true){//state=0メロディ入力状態時の確認画面表示
			int xsize = 0, x = 660/llm.size(); 
			for(int i = 0; i < llm.size(); i++){//記録したメロディ情報追加
				wx.fillRect(xsize, (300 - (llm.get(i) - 48) * 5 ), x, 4);
				wx.fillRect(xsize, (304 - (llm.get(i) - 48) * 5 ), 4, 2);//この２つの式は音の出だしを表す
				wx.fillRect(xsize, (298 - (llm.get(i) - 48) * 5 ), 4, 2);
				xsize += x;
			}
		}
		if(state==1 || state ==2){//state!=0メロディ入力時以外の確認画面表示
			double ox4s, ox4;
			divx = 1;
			xpos.clear();
			ypos.clear();
			while((double)((llt.get((llt.size()-1)) + llr.get((llt.size()-1))) / divx) > 660)//枠内に収まるようにtickを割るxを決定
				divx = divx + 0.1; 
			for(int i = 0; i < llt.size(); i++){//記録したリズム情報追加
				wx.fillRect((int)(llt.get(i) / divx), (300 - (llm.get(i) - 48) * 5 ), (int)(llr.get(i) / divx), 4);
				xpos.add((int)(llt.get(i) / divx));//xpos偶数場所に音の始まりのx座標
				xpos.add((int)(llr.get(i) / divx));//xpos奇数場所に加算分のx座標
				ypos.add(300 - (llm.get(i) - 48) * 5);
			}
			ox4s = 60000/temposcr.getValue()/divx;//4分音符ごとに縦線
			ox4 = ox4s;
			while(ox4 < 660){
				wx.drawLine((int)ox4,120,(int)ox4,304);//縦線処理
				ox4 += ox4s;
			}
		}
	}
	public void paint(Graphics g) {
		g.drawImage(img,20,30,this); //イメージの描写
		g.drawImage(img2,700,130,this); //メトロノームイメージの描写
	}
	//ボタンの処理関数
	public void actionPerformed(ActionEvent e){
		if(e.getSource()==play){//再生停止
			if(seqer.isRunning() == true)
				musicStop();
			else{
				try{
					if(state==0){//メロディ入力の場合
						seqer.setSequence(seq);//メロディシーケンスseqを設定
						seqer.setTickPosition(0);// Sequencerを初期位置に戻す
						seqer.setTempoInMPQ(seq.getResolution() * 1000);//シーケンスで決定されたテンポをシーケンサへマイクロ秒なので1000倍
						//MidiSystem.write(seq,0,new java.io.File("C:\\Documents and Settings\\kizaki\\デスクトップ\\search2\\midi2\\kakidasi.mid"));//書き出し
					}
					else if(state==1 || state==2){//リズム入力の場合
						seqer.setSequence(seq2);//リズムシーケンスseq2を設定
						seqer.setTickPosition(0);// Sequencerを初期位置に戻す
						seqer.setTempoInMPQ(seq2.getResolution() * 1000);//シーケンスで決定されたテンポをシーケンサへマイクロ秒なので1000倍
						//MidiSystem.write(seq,0,new java.io.File("C:\\Documents and Settings\\kizaki\\デスクトップ\\search2\\midi2\\kakidasi2.mid"));
					}
					seqer.start();// Sequencer開始
				}catch  (Exception er) {
					er.printStackTrace();
				}
			}
		}
		else if(e.getSource()==play2){//検索曲の再生
			if(seqer2.isRunning() == true)
				musicStop();
			else{
				seqer2.setTickPosition(new Long(tickla.getText()).longValue()); 
				seqer2.start();
			}
		}
		else{
			musicStop();//再生中に他のボタン操作を行った場合に再生停止
			if(e.getSource()==bs){//最後尾音削除処理
 				if(state==0){//melody入力時
					if(llm.size() != 0){
						llm.removeLast();//llmの最後尾削除
						try{//noteoffから先に消去する
							track.remove(track.get(2*(llm.size())+1));//noteoff削除
							track.remove(track.get(2*(llm.size())));//noteon削除
						}catch  (Exception er) {
							er.printStackTrace();
						}
						mtime -= 500;//削除した分だけメロディタイムを減らす(500)
						if(llm.size() == 0)//もし削除することによってllmが空になったらncon=false
							ncon = false;
						disp();
						repaint();
					}
				}
				expla.setText(m[3]);
			}
			else if(e.getSource()==re){//reset段階に応じて1段階前の状態に戻す
				if(state==0){//メロディ入力状態の場合
					mtime = 0;
					llm.clear();//メロディ情報破棄
					seq.deleteTrack(track);//今までのメロディ入力を破棄
					track = seq.createTrack();//シーケンスに空のトラックを
					ncon = false;
				}
				else if(state==1){//リズム入力状態
					llr.clear();
					llt.clear();
					seq2.deleteTrack(track2);//リズムが入ったtrackをdelete
					re.setLabel("melody reset");
					state = 0;//メロディにシフト
				}
				else{//correct状態state=2
					llt.clear();
					seq2.deleteTrack(track2);//リズムが入ったtrackをdelete
					track2 = seq2.createTrack();//シーケンスに空のトラックを
					for(int i = 0;i< lltc.size();i++){//補正前のtick情報を渡す
						llt.add(lltc.get(i));
						addNoteDate(track2, llm.get(i), llt.get(i), llr.get(i));
					}
					re.setLabel("rhythm reset");
					state = 1;
				}
				expla.setText(m[4]);
				disp();
				repaint();
			}
			else if(e.getSource()==crt){//補正
				//String test="";
				int crttempo, crtf=0;//補正後のtempo(補正幅)int
				long crttick;//補正後のtick
				if(state==1 || state==2){//リズム状態又は補正状態
					if(state == 1){//tickのコピー
						lltc.clear();
						for(int i=0;i<llt.size();i++)
							lltc.add(llt.get(i));
					}
					if(correct.getSelectedIndex() == 0){//補正の幅を補正スクロールバーから求める
						crttempo = Math.round(60000/temposcr.getValue());
						crtf = 1;
					}
					else if(correct.getSelectedIndex() == 1){
						crttempo = Math.round(60000/temposcr.getValue()/2);
						crtf = 2;
					}
					else{
						crttempo = Math.round(60000/temposcr.getValue()/4);
						crtf = 4;
					}
					//test = crttempo + ": ";
					for(int i = 1; i < llt.size()-1; i++){//最初のtickは0で固定それ以上は最後以外補正
						crttick = Math.round((double)llt.get(i)/crttempo) * crttempo;//補正したら重なる場合は一番近い部分を補正
						if(Math.abs(crttick - llt.get(i)) < Math.abs(crttick - llt.get(i-1)) &&
						Math.abs(crttick - llt.get(i)) < Math.abs(crttick - llt.get(i+1))){
							llt.set(i,crttick);
							//test += crttick;
							//test += " ";
						}
					}
					if(llt.size() >= 2){
						crttick = Math.round((double)llt.get(llt.size()-1)/crttempo) * crttempo;//最後の部分の補正
						if(Math.abs(crttick - llt.get(llt.size()-2)) > Math.abs(crttick - llt.get(llt.size()-1)))
							llt.set(llt.size()-1,crttick);
					}
					try{
						seq2 = new Sequence(Sequence.PPQ,crttempo*crtf);//設定したtempoを代入
					}catch  (Exception er) {
						er.printStackTrace();
					}
					track2 = seq2.createTrack();
					for(int i = 0;i< llt.size();i++)//補正したtickのリストをシーケンスへ
						addNoteDate(track2, llm.get(i), llt.get(i), llr.get(i));
					state = 2;//correct状態へ
					re.setLabel("correct reset");
					disp();
					repaint();
					expla.setText(/*test*/m[5]);
				}
			}
			else if(e.getSource()==sch){//検索
				String[] args = {"C:\\Documents and Settings\\kizaki\\デスクトップ\\search2\\midi2\\multiChannel","",""};
				args[2] = diff2.getText();
				for(int i=0;i<llm.size();i++){
					if(state == 0){
						args[1] += llm.get(i);
						args[1] += " ";
					}
					else{//PPQを480にあわせる処理
						double tempodiv = (double)seq2.getResolution()/480;
						if(i!=0){
							//args[1] += Math.round((Math.round((llt.get(i)-llt.get(i-1))/tempodiv) / 10) * 10);//デルタタイム
							args[1] += Math.round((llt.get(i)-llt.get(i-1))/tempodiv);
							args[1] += " ";
						}
						args[1] += llm.get(i);
						args[1] += " ";
					}
				}
				expla.setText(args[1]);
				try{
					
					MidiSystem.write(seq,0,new java.io.File("C:\\Documents and Settings\\kizaki\\デスクトップ\\search2\\midi2\\kakidasi.mid"));//書き出し
					MidiSystem.write(seq2,0,new java.io.File("C:\\Documents and Settings\\kizaki\\デスクトップ\\search2\\midi2\\kakidasi2.mid"));//書き出し
				}catch  (Exception er) {
					er.printStackTrace();
				}
				start(args);
			}
			else if(e.getSource()==hitlist){//検索曲の選択
				String[] music=(hitlist.getSelectedItem()).split(" : ");
				playdate = music[0];
				playtick= new Long(music[1]);
				try{
					seqer2.setSequence(MidiSystem.getSequence(new java.io.File(playdate))); 
				}catch  (Exception er) {
					er.printStackTrace();
				}
				tickla.setText(music[1]);
				tickscr.setValues(playtick.intValue(),100,0,(int)seqer2.getTickLength());
				repaint();
			}
		}
	}
	//マウス処理の関数
	public void mousePressed(MouseEvent e){
		if(e.getSource()==rtm){//リズムボタンが入力されたら
			musicStop();//再生中だったら停止
			if(state==0 &&ncon==true){//メロディ入力からだったらメロディシーケンスからリズムシーケンスへの切り替え
				try{
					seq2 = new Sequence(Sequence.PPQ,60000/temposcr.getValue());//実際は設定したテンポを代入
				}catch  (Exception er) {
					er.printStackTrace();
				}
				track2 = seq2.createTrack();
				allstart = System.currentTimeMillis();//１番最初の時間を保持
				ir = 0;//メロディの音位置を初期位置へ
				state=1;//リズム入力にシフト
				re.setLabel("rhythm reset");
			}
			if(llm.size() > ir){//メロディの長さを超えていなければ
				note = llm.get(ir);
				chan[0].noteOn(note,127);
				start = System.currentTimeMillis();
			}
		}
		else{//メロディ入力
			int x = e.getX(), y = e.getY();
			note = getnote(x,y);
			if(note!=0){//鍵盤を入力したときのみ
				musicStop();
				if(state==1||state==2){//リズム入力からだったら
 					state=0;//メロディ入力にシフト
					llm.clear();//前回のメロディ記録を削除
					llr.clear();//前回の音長記録を削除
					llt.clear();//前回の音の出だし記録を削除
					mtime = 0; //メロディタイムを初期に戻す
					seq.deleteTrack(track);//前回のメロディ入力のトラック削除
					track = seq.createTrack();
					re.setLabel("melody reset");
				}
				chan[0].noteOn(note,127);
				addNoteDate(track, note, mtime, 500);
				mtime += 500;
				for(int i = 0; i < t.length; i++){//押した鍵盤を赤くする処理
					if(note == Integer.parseInt(t[i][0])){
						wx.setColor(Color.red);
						if(t[i][2] == "0")
							wx.fillRect(Integer.parseInt(t[i][1]),50,28,50);//白鍵を塗りつぶす
						else
							wx.fillRect(Integer.parseInt(t[i][1]),0,28,50);//黒鍵を塗りつぶす
					}
				}
				llm.add(note);//メロディ情報追加
				ncon = true;//メロディ情報が蓄積された
				expla.setText(m[1]);
				repaint();
			}
			else if(x>20&&x<680&&y>150&&y<334){//マウスによる確認画面のnote選択
				if(state != 0){//メロディ入力以外なら
					for(int i=2;i<xpos.size()-1;i=i+2){//選択するnoteの検索
						if(x>xpos.get(i)+20&&x<xpos.get(i)+xpos.get(i+1)+20){
							slc = i/2;//slcへ選択notenumberを渡す
							wx.setColor(Color.green);
							wx.fillRect(xpos.get(i), ypos.get(i/2),xpos.get(i+1),4);//選択noteを緑へ
							break;
						}
					}
				}
				repaint();
			}
		}
	}
	public void mouseReleased(MouseEvent e){
		if(e.getSource()==rtm){//リズム入力からだったら
			if(llm.size() > ir){
				ir++;
				chan[0].noteOff(note,0);
				stop = System.currentTimeMillis();
				time = stop - start;
				llr.add(time);//音長の保持
				ttime = start - allstart;
				llt.add(ttime);//音の出だしの保持
				addNoteDate(track2, note, ttime, time);
				expla.setText(m[2]);
				disp();
				repaint();
			}
		}
		else if(state == 0){//メロディ入力からだったら
			chan[0].noteOff(note,0);
			disp();
			repaint();
		}
		else if(slc != 0){//選択nodeの移動
			long diff;
			int x = e.getX()-20;
			diff =  (long)(x * divx) - llt.get(slc);//diff=移動した分の差
			if(llt.get(slc-1) < llt.get(slc)+diff){//移動する場所が前のnoteより小さくならないのであれば移動
				for(int i=slc; i<llt.size();i++)
					llt.set(i,llt.get(i)+diff);
				try{
					seq2 = new Sequence(Sequence.PPQ,60000/temposcr.getValue());//設定したtempoを代入
				}catch  (Exception er) {
					er.printStackTrace();
				}
				track2 = seq2.createTrack();
				for(int i = 0;i< llt.size();i++)//補正したtickのリストをシーケンスへ
					addNoteDate(track2, llm.get(i), llt.get(i), llr.get(i));
			}
			disp();
			repaint();
			slc = 0;
		}
	}
	public void mouseClicked(MouseEvent e){
	}
	public void mouseEntered(MouseEvent e){
	}
	public void mouseExited(MouseEvent e){
	}
	//スクロールバーの関数
	public void adjustmentValueChanged(AdjustmentEvent e){
		if(e.getSource() == temposcr)
			tempola.setText(tempost.valueOf(temposcr.getValue()));//スクロールバーとtempo表示textの関連付け
		else if(e.getSource() == tickscr)
			tickla.setText(new String().valueOf(tickscr.getValue()));
		disp();
		repaint();
	}
	//programをまとめるための関数
	public void set(Component com, int x, int y, int sizex, int sizey){//コンポーネントの配置
		com.setLocation(x,y);
		com.setSize(sizex,sizey);
	}
	public int getnote(int x, int y){//出力すべき音の入手する関数
		int n = 0;
		for(int i = 0; i < t.length; i++){
			if(t[i][2] == "0"){//白鍵
				if(x > 20+Integer.parseInt(t[i][1]) && x < 20+Integer.parseInt(t[i][1])+28){
					if(y > 30 && y < 130){
						n = Integer.parseInt(t[i][0]);
						break;
					}
				}
			}
		}
		for(int i = 0; i < t.length; i++){
			if(t[i][2] == "1"){//黒鍵
				if(x > 20+Integer.parseInt(t[i][1]) && x < 20+Integer.parseInt(t[i][1])+28){
					if(y > 30 && y < 80){
						n = Integer.parseInt(t[i][0]);
						break;
					}
				}
			}
		}
		return n;
	}
	public void addNoteDate(Track track, int note, long tick, long addtick){//note情報をトラックに加える
		try{
			ShortMessage mes = new ShortMessage(); //音の開始ショートメッセージ
			mes.setMessage(ShortMessage.NOTE_ON, 0, note, 127);
			track.add(new MidiEvent(mes,tick));
			ShortMessage mesf = new ShortMessage();//音の終了ショートメッセージ
			mesf.setMessage(ShortMessage.NOTE_OFF, 0, note, 127);
			track.add(new MidiEvent(mesf,tick+addtick));
		}catch(Exception er) {
			er.printStackTrace();
		}
	}
	public void musicStop(){//シーケンサ停止
		if(seqer.isRunning() == true)
			seqer.stop();//Sequencer停止
		if(seqer2.isRunning() == true)
			seqer2.stop();//Sequencer停止
	}
}
