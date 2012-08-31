import java.applet.Applet;
import java.util.*;
import java.awt.*;
import java.awt.event.*;
import javax.sound.midi.*;


//�C���^�[�t�F�C�X���C��
//����^
//main10�ɉ����ă��g���m�[�������̕ύX
//��B�H�Ƒ�w���H�w�� �m�\���H�w�� 03231028 �؍� �M�v
/*
	<applet code="MainDisplay2" width = 820 height = 700>
	</applet>
*/
public class MainDisplay2 extends Applet implements ActionListener,Runnable,MouseListener,AdjustmentListener{
	LinkedList<Integer> llm = new LinkedList<Integer>();//��������ێ�����LinkedList
	LinkedList<Long> llr=new LinkedList<Long>(),llt=new LinkedList<Long>(),lltc=new LinkedList<Long>();//llr-����llt-tick���lltc-tick�̃R�s�[
	ArrayList<Integer> xpos = new ArrayList<Integer>(), ypos = new ArrayList<Integer>();//xpos,ypos=�m�F��ʂ�note��x���W�̃��X�gnote�ړ��ɕK�v
	Synthesizer synth = null;//�V���Z�T�C�U�[�̍쐬
	MidiChannel chan[] = null;//midi�`���l���̍쐬
	Track track = null,track2 = null;//track�̍쐬 track�̓����f�Btrack track2�̓��Y��track
	Sequencer seqer = null, seqer2=null;//�V�[�P���T�̍쐬
	Sequence seq = null, seq2 = null;//seq�����f�B�̃V�[�P���X seq2���Y���̃V�[�P���X
	Image img,img2;//img2�̓��g���m�[������  img�͂���ȊO(���Պm�F���)�̃C���[�W
	Graphics wx,wx2;//Graphics�̍쐬 wx�͌��� ���o�͕��� wx2�̓��g���m�[������
	String tempost;//�e���|��String�\��
	TextField diff2;//�����ɑ΂��鑍tick�̂���
	int note, ir = 0, state = 0, slc=0;//note=���� ir=���Y�����͎��̉����̌��݈ʒu state=��ԃ����f�B�P���Y���Q�␳�R slc=�m�F��ʏ�̑I��notenumber
	double divx = 1;//�m�F��ʂ̕����l
	long allstart, start, stop, time, ttime, mtime= 0;//time=���Y���̓��͎��� ttime=�e�B�b�N mtime=�����f�B�^�C��500(8������)����
	boolean ncon = false;//ncon=true�Ȃ�llm�ɗv�f�������Ă���
	Button play,rtm,bs,re,crt,sch,play2;//play=�Đ���~ rtm=���Y������ bs=noteBS re=reset crt=correct sch=search
	java.awt.List correct, hitlist; //�␳�͈͂����肷��list
	Scrollbar temposcr, tickscr; //tempo��ݒ肷��X�N���[���o�[ 
	Thread th, thread1;//th���g���m�[���@�\�ׂ̈̃X���b�h thread1
	Label expla,tempola,tickla; //���x��(�������) tempotf//tempo�̌��ݒl
	String  m[] = {"���Ձ�search��melody���� rhythm����rhythm��correct ��search��rhythm����",
				"���Ղ�melody(����)�̓��݂͂̂ŉ����͓���Ȃ��Brhythm(����)����͂���ɂ�rhythmbutton��",
				"���\����ʂ̏c����4���������݁B�ŏ���rhythm����͂��Ă���Ō��rhythm����͂���܂Ń��A���^�C���Ői��",
				"melody���͎��̂ݍŌ���̉���1����������",
				"���i�K�̏���j������1�i�K�O�̏�Ԃ֑J�� melody�������� rhythm��melody correct��rhythm",
				"List�őI��(4or8or16)�����ɂ���ĕ␳�ŏ�����������.���ꂼ��̕␳������1�ԋ߂����݂̂�␳",
				"���݂͓��͂�������TextField�֕\��"}, playdate;
	//�L�[���P�D���K�Q�D�ʒu�R�D��������
	String t[][] = {{"48","000","0"},{"49","015","1"},{"50","030","0"},{"51","045","1"},{"52","060","0"},
					{"53","090","0"},{"54","105","1"},{"55","120","0"},{"56","135","1"},{"57","150","0"},
					{"58","165","1"},{"59","180","0"},{"60","210","0"},{"61","225","1"},{"62","240","0"},
					{"63","255","1"},{"64","270","0"},{"65","300","0"},{"66","315","1"},{"67","330","0"},
					{"68","345","1"},{"69","360","0"},{"70","375","1"},{"71","390","0"},{"72","420","0"},
					{"73","435","1"},{"74","450","0"},{"75","465","1"},{"76","480","0"},{"77","510","0"},
					{"78","525","1"},{"79","540","0"},{"80","555","1"},{"81","570","0"},{"82","585","1"},
					{"83","600","0"},{"84","630","0"}};
	Long playtick;//�Đ��ꏊ��tick
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
			synth = MidiSystem.getSynthesizer();// �V���Z�T�C�U�[���擾
			chan = synth.getChannels();// MIDI�`�����l�����X�g���擾
			seq = new Sequence(Sequence.PPQ,1000);//PPQ�͎l�������������tick �����f�B�V�[�P���X�̍쐬
			track = seq.createTrack();//��̃g���b�N�쐬�ɂ̓R���X�g���N�^�ł͂Ȃ��E�̎������s
			seqer = MidiSystem.getSequencer();//�f�t�H���g�̃V�[�P���T���擾
			seqer2= MidiSystem.getSequencer();
			seqer.open();//�V�[�P���T���J��
			seqer2.open();
			synth.open();// �V���Z�T�C�U�[���J��
		}catch  (Exception e) {
			e.printStackTrace();
		}
		img  = createImage(680,400);
		wx  = img.getGraphics();
		img2  = createImage(101,20); 
		wx2  = img2.getGraphics();
		setLayout(null);//�R���|�[�l���g�̔z�u
		
		add(expla = new Label());//����Label�̒ǉ�
		set(expla,5,2,810,18);
		expla.setText(m[0]);
		
		add(rtm=new Button("rhythm"));//���Y���{�^���̒ǉ�
		set(rtm,700,30,100,100);
		rtm.addMouseListener(this);
		
		add(temposcr = new Scrollbar(Scrollbar.HORIZONTAL,60,10,20,210));//�e���|�X�N���[���o�[�͈̓��g���m�[��20~200�܂�
		set(temposcr,700,150,100,15);
		temposcr.addAdjustmentListener(this);
		
		add(tempola=new Label());//�e���|�\�����x���̒ǉ�
		set(tempola,700,165,40,20);
		tempola.setText(tempost.valueOf(temposcr.getValue()));
		
		add(correct=new java.awt.List(1,false));//�␳�͈̓��x���̒ǉ�
		correct.add("4��");correct.add("8��");correct.add("16��");
		correct.select(0);
		correct.makeVisible(0);
		set(correct,750,165,50,20);
		
		add(crt=new Button("correct"));//correct�̒ǉ�
		set(crt,700,185,100,25);
		crt.addActionListener(this);
		
		add(bs=new Button("note backspace"));//noteBS�̒ǉ�
		set(bs,700,215,100,25);
		bs.addActionListener(this);
		
		add(re=new Button("melody reset"));//reset�̒ǉ�
		set(re,700,240,100,25);
		re.addActionListener(this);
		
		add(play=new Button("play/stop"));//�Đ�/��~�{�^���̒ǉ�
		set(play,700,280,100,25);
		play.addActionListener(this);
		
		add(sch=new Button("search"));//�����{�^���̒ǉ�
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
		
		addMouseListener(this);//�A�v���b�g�̃N���b�N�ɑ΂���C�x���g�̒ǉ�
		disp();
		th = new Thread(this);//�X���b�h�̎g�p
		th.start();//�X���b�h�̊J�n
	}
	public void run(){//�X���b�h���g�p���ă��g���m�[���@�\������B�_�ł̊Ԋu�͂W�������
		try{
			while(true){
				for(int x=50;x<100;x=x+25){
					wx2.setColor(Color.white);
					wx2.fillRect(0,0,101,20);
					wx2.setColor(Color.black);
					wx2.drawLine(50,0,50,20);//���g���m�[�����
					wx2.setColor(Color.blue);
					wx2.drawLine(x,0,x,20);
					repaint();
					Thread.sleep(60000/temposcr.getValue()/8);//�e���|���~���b�֕ϊ�
				}
				for(int x=100;x>50;x=x-25){
					wx2.setColor(Color.white);
					wx2.fillRect(0,0,101,20);
					wx2.setColor(Color.black);
					wx2.drawLine(50,0,50,20);//���g���m�[�����
					wx2.setColor(Color.blue);
					wx2.drawLine(x,0,x,20);
					repaint();
					Thread.sleep(60000/temposcr.getValue()/8); 
				}
				for(int x=50;x>0;x=x-25){
					wx2.setColor(Color.white);
					wx2.fillRect(0,0,101,20);
					wx2.setColor(Color.black);
					wx2.drawLine(50,0,50,20);//���g���m�[�����
					wx2.setColor(Color.blue);
					wx2.drawLine(x,0,x,20);
					repaint();
					Thread.sleep(60000/temposcr.getValue()/8); 
				}
				for(int x=0;x<50;x=x+25){
					wx2.setColor(Color.white);
					wx2.fillRect(0,0,101,20);
					wx2.setColor(Color.black);
					wx2.drawLine(50,0,50,20);//���g���m�[�����
					wx2.setColor(Color.blue);
					wx2.drawLine(x,0,x,20);
					repaint();
					Thread.sleep(60000/temposcr.getValue()/8); 
				}
			}
		}catch(Exception e){
		}
	}
	public void update(Graphics g){//��������������Ȃ��悤�ɂ���update���\�b�h�̃I�[�o�[���C�h
		paint(g);
	}
	public void disp(){//��ʂ��\������֐�
		setBackground(Color.lightGray);
		wx.setColor(Color.lightGray);
		wx.fillRect(0,0,800,400);//�S�ʂ�lightGray�œh��Ԃ�
		for (int i=0;i<t.length;i++) {
			if (t[i][2]=="0") { //����
				wx.setColor(Color.white); 
				wx.fillRect(Integer.parseInt(t[i][1]),0,28,100);//������h��Ԃ�
			}
		}
		for (int i=0;i<t.length;i++) {
			if (t[i][2]=="1") {//����
				wx.setColor(Color.black); 
				wx.fillRect(Integer.parseInt(t[i][1]),0,28,50);
			}
		}
		//��������m�F��ʂ��쐬
		wx.setColor(Color.white);
		wx.fillRect(0,120,660,184); //�g�̍쐬��660�c184
		wx.setColor(Color.black);
		//����
		wx.drawLine(0,180,660,180);
		wx.drawLine(0,240,660,240); //��������C�̊��
		wx.setColor(Color.red);
		if(state==0&&ncon==true){//state=0�����f�B���͏�Ԏ��̊m�F��ʕ\��
			int xsize = 0, x = 660/llm.size(); 
			for(int i = 0; i < llm.size(); i++){//�L�^���������f�B���ǉ�
				wx.fillRect(xsize, (300 - (llm.get(i) - 48) * 5 ), x, 4);
				wx.fillRect(xsize, (304 - (llm.get(i) - 48) * 5 ), 4, 2);//���̂Q�̎��͉��̏o������\��
				wx.fillRect(xsize, (298 - (llm.get(i) - 48) * 5 ), 4, 2);
				xsize += x;
			}
		}
		if(state==1 || state ==2){//state!=0�����f�B���͎��ȊO�̊m�F��ʕ\��
			double ox4s, ox4;
			divx = 1;
			xpos.clear();
			ypos.clear();
			while((double)((llt.get((llt.size()-1)) + llr.get((llt.size()-1))) / divx) > 660)//�g���Ɏ��܂�悤��tick������x������
				divx = divx + 0.1; 
			for(int i = 0; i < llt.size(); i++){//�L�^�������Y�����ǉ�
				wx.fillRect((int)(llt.get(i) / divx), (300 - (llm.get(i) - 48) * 5 ), (int)(llr.get(i) / divx), 4);
				xpos.add((int)(llt.get(i) / divx));//xpos�����ꏊ�ɉ��̎n�܂��x���W
				xpos.add((int)(llr.get(i) / divx));//xpos��ꏊ�ɉ��Z����x���W
				ypos.add(300 - (llm.get(i) - 48) * 5);
			}
			ox4s = 60000/temposcr.getValue()/divx;//4���������Ƃɏc��
			ox4 = ox4s;
			while(ox4 < 660){
				wx.drawLine((int)ox4,120,(int)ox4,304);//�c������
				ox4 += ox4s;
			}
		}
	}
	public void paint(Graphics g) {
		g.drawImage(img,20,30,this); //�C���[�W�̕`��
		g.drawImage(img2,700,130,this); //���g���m�[���C���[�W�̕`��
	}
	//�{�^���̏����֐�
	public void actionPerformed(ActionEvent e){
		if(e.getSource()==play){//�Đ���~
			if(seqer.isRunning() == true)
				musicStop();
			else{
				try{
					if(state==0){//�����f�B���͂̏ꍇ
						seqer.setSequence(seq);//�����f�B�V�[�P���Xseq��ݒ�
						seqer.setTickPosition(0);// Sequencer�������ʒu�ɖ߂�
						seqer.setTempoInMPQ(seq.getResolution() * 1000);//�V�[�P���X�Ō��肳�ꂽ�e���|���V�[�P���T�փ}�C�N���b�Ȃ̂�1000�{
						//MidiSystem.write(seq,0,new java.io.File("C:\\Documents and Settings\\kizaki\\�f�X�N�g�b�v\\search2\\midi2\\kakidasi.mid"));//�����o��
					}
					else if(state==1 || state==2){//���Y�����͂̏ꍇ
						seqer.setSequence(seq2);//���Y���V�[�P���Xseq2��ݒ�
						seqer.setTickPosition(0);// Sequencer�������ʒu�ɖ߂�
						seqer.setTempoInMPQ(seq2.getResolution() * 1000);//�V�[�P���X�Ō��肳�ꂽ�e���|���V�[�P���T�փ}�C�N���b�Ȃ̂�1000�{
						//MidiSystem.write(seq,0,new java.io.File("C:\\Documents and Settings\\kizaki\\�f�X�N�g�b�v\\search2\\midi2\\kakidasi2.mid"));
					}
					seqer.start();// Sequencer�J�n
				}catch  (Exception er) {
					er.printStackTrace();
				}
			}
		}
		else if(e.getSource()==play2){//�����Ȃ̍Đ�
			if(seqer2.isRunning() == true)
				musicStop();
			else{
				seqer2.setTickPosition(new Long(tickla.getText()).longValue()); 
				seqer2.start();
			}
		}
		else{
			musicStop();//�Đ����ɑ��̃{�^��������s�����ꍇ�ɍĐ���~
			if(e.getSource()==bs){//�Ō�����폜����
 				if(state==0){//melody���͎�
					if(llm.size() != 0){
						llm.removeLast();//llm�̍Ō���폜
						try{//noteoff�����ɏ�������
							track.remove(track.get(2*(llm.size())+1));//noteoff�폜
							track.remove(track.get(2*(llm.size())));//noteon�폜
						}catch  (Exception er) {
							er.printStackTrace();
						}
						mtime -= 500;//�폜���������������f�B�^�C�������炷(500)
						if(llm.size() == 0)//�����폜���邱�Ƃɂ����llm����ɂȂ�����ncon=false
							ncon = false;
						disp();
						repaint();
					}
				}
				expla.setText(m[3]);
			}
			else if(e.getSource()==re){//reset�i�K�ɉ�����1�i�K�O�̏�Ԃɖ߂�
				if(state==0){//�����f�B���͏�Ԃ̏ꍇ
					mtime = 0;
					llm.clear();//�����f�B���j��
					seq.deleteTrack(track);//���܂ł̃����f�B���͂�j��
					track = seq.createTrack();//�V�[�P���X�ɋ�̃g���b�N��
					ncon = false;
				}
				else if(state==1){//���Y�����͏��
					llr.clear();
					llt.clear();
					seq2.deleteTrack(track2);//���Y����������track��delete
					re.setLabel("melody reset");
					state = 0;//�����f�B�ɃV�t�g
				}
				else{//correct���state=2
					llt.clear();
					seq2.deleteTrack(track2);//���Y����������track��delete
					track2 = seq2.createTrack();//�V�[�P���X�ɋ�̃g���b�N��
					for(int i = 0;i< lltc.size();i++){//�␳�O��tick����n��
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
			else if(e.getSource()==crt){//�␳
				//String test="";
				int crttempo, crtf=0;//�␳���tempo(�␳��)int
				long crttick;//�␳���tick
				if(state==1 || state==2){//���Y����Ԗ��͕␳���
					if(state == 1){//tick�̃R�s�[
						lltc.clear();
						for(int i=0;i<llt.size();i++)
							lltc.add(llt.get(i));
					}
					if(correct.getSelectedIndex() == 0){//�␳�̕���␳�X�N���[���o�[���狁�߂�
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
					for(int i = 1; i < llt.size()-1; i++){//�ŏ���tick��0�ŌŒ肻��ȏ�͍Ō�ȊO�␳
						crttick = Math.round((double)llt.get(i)/crttempo) * crttempo;//�␳������d�Ȃ�ꍇ�͈�ԋ߂�������␳
						if(Math.abs(crttick - llt.get(i)) < Math.abs(crttick - llt.get(i-1)) &&
						Math.abs(crttick - llt.get(i)) < Math.abs(crttick - llt.get(i+1))){
							llt.set(i,crttick);
							//test += crttick;
							//test += " ";
						}
					}
					if(llt.size() >= 2){
						crttick = Math.round((double)llt.get(llt.size()-1)/crttempo) * crttempo;//�Ō�̕����̕␳
						if(Math.abs(crttick - llt.get(llt.size()-2)) > Math.abs(crttick - llt.get(llt.size()-1)))
							llt.set(llt.size()-1,crttick);
					}
					try{
						seq2 = new Sequence(Sequence.PPQ,crttempo*crtf);//�ݒ肵��tempo����
					}catch  (Exception er) {
						er.printStackTrace();
					}
					track2 = seq2.createTrack();
					for(int i = 0;i< llt.size();i++)//�␳����tick�̃��X�g���V�[�P���X��
						addNoteDate(track2, llm.get(i), llt.get(i), llr.get(i));
					state = 2;//correct��Ԃ�
					re.setLabel("correct reset");
					disp();
					repaint();
					expla.setText(/*test*/m[5]);
				}
			}
			else if(e.getSource()==sch){//����
				String[] args = {"C:\\Documents and Settings\\kizaki\\�f�X�N�g�b�v\\search2\\midi2\\multiChannel","",""};
				args[2] = diff2.getText();
				for(int i=0;i<llm.size();i++){
					if(state == 0){
						args[1] += llm.get(i);
						args[1] += " ";
					}
					else{//PPQ��480�ɂ��킹�鏈��
						double tempodiv = (double)seq2.getResolution()/480;
						if(i!=0){
							//args[1] += Math.round((Math.round((llt.get(i)-llt.get(i-1))/tempodiv) / 10) * 10);//�f���^�^�C��
							args[1] += Math.round((llt.get(i)-llt.get(i-1))/tempodiv);
							args[1] += " ";
						}
						args[1] += llm.get(i);
						args[1] += " ";
					}
				}
				expla.setText(args[1]);
				try{
					
					MidiSystem.write(seq,0,new java.io.File("C:\\Documents and Settings\\kizaki\\�f�X�N�g�b�v\\search2\\midi2\\kakidasi.mid"));//�����o��
					MidiSystem.write(seq2,0,new java.io.File("C:\\Documents and Settings\\kizaki\\�f�X�N�g�b�v\\search2\\midi2\\kakidasi2.mid"));//�����o��
				}catch  (Exception er) {
					er.printStackTrace();
				}
				start(args);
			}
			else if(e.getSource()==hitlist){//�����Ȃ̑I��
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
	//�}�E�X�����̊֐�
	public void mousePressed(MouseEvent e){
		if(e.getSource()==rtm){//���Y���{�^�������͂��ꂽ��
			musicStop();//�Đ������������~
			if(state==0 &&ncon==true){//�����f�B���͂��炾�����烁���f�B�V�[�P���X���烊�Y���V�[�P���X�ւ̐؂�ւ�
				try{
					seq2 = new Sequence(Sequence.PPQ,60000/temposcr.getValue());//���ۂ͐ݒ肵���e���|����
				}catch  (Exception er) {
					er.printStackTrace();
				}
				track2 = seq2.createTrack();
				allstart = System.currentTimeMillis();//�P�ԍŏ��̎��Ԃ�ێ�
				ir = 0;//�����f�B�̉��ʒu�������ʒu��
				state=1;//���Y�����͂ɃV�t�g
				re.setLabel("rhythm reset");
			}
			if(llm.size() > ir){//�����f�B�̒����𒴂��Ă��Ȃ����
				note = llm.get(ir);
				chan[0].noteOn(note,127);
				start = System.currentTimeMillis();
			}
		}
		else{//�����f�B����
			int x = e.getX(), y = e.getY();
			note = getnote(x,y);
			if(note!=0){//���Ղ���͂����Ƃ��̂�
				musicStop();
				if(state==1||state==2){//���Y�����͂��炾������
 					state=0;//�����f�B���͂ɃV�t�g
					llm.clear();//�O��̃����f�B�L�^���폜
					llr.clear();//�O��̉����L�^���폜
					llt.clear();//�O��̉��̏o�����L�^���폜
					mtime = 0; //�����f�B�^�C���������ɖ߂�
					seq.deleteTrack(track);//�O��̃����f�B���͂̃g���b�N�폜
					track = seq.createTrack();
					re.setLabel("melody reset");
				}
				chan[0].noteOn(note,127);
				addNoteDate(track, note, mtime, 500);
				mtime += 500;
				for(int i = 0; i < t.length; i++){//���������Ղ�Ԃ����鏈��
					if(note == Integer.parseInt(t[i][0])){
						wx.setColor(Color.red);
						if(t[i][2] == "0")
							wx.fillRect(Integer.parseInt(t[i][1]),50,28,50);//������h��Ԃ�
						else
							wx.fillRect(Integer.parseInt(t[i][1]),0,28,50);//������h��Ԃ�
					}
				}
				llm.add(note);//�����f�B���ǉ�
				ncon = true;//�����f�B��񂪒~�ς��ꂽ
				expla.setText(m[1]);
				repaint();
			}
			else if(x>20&&x<680&&y>150&&y<334){//�}�E�X�ɂ��m�F��ʂ�note�I��
				if(state != 0){//�����f�B���͈ȊO�Ȃ�
					for(int i=2;i<xpos.size()-1;i=i+2){//�I������note�̌���
						if(x>xpos.get(i)+20&&x<xpos.get(i)+xpos.get(i+1)+20){
							slc = i/2;//slc�֑I��notenumber��n��
							wx.setColor(Color.green);
							wx.fillRect(xpos.get(i), ypos.get(i/2),xpos.get(i+1),4);//�I��note��΂�
							break;
						}
					}
				}
				repaint();
			}
		}
	}
	public void mouseReleased(MouseEvent e){
		if(e.getSource()==rtm){//���Y�����͂��炾������
			if(llm.size() > ir){
				ir++;
				chan[0].noteOff(note,0);
				stop = System.currentTimeMillis();
				time = stop - start;
				llr.add(time);//�����̕ێ�
				ttime = start - allstart;
				llt.add(ttime);//���̏o�����̕ێ�
				addNoteDate(track2, note, ttime, time);
				expla.setText(m[2]);
				disp();
				repaint();
			}
		}
		else if(state == 0){//�����f�B���͂��炾������
			chan[0].noteOff(note,0);
			disp();
			repaint();
		}
		else if(slc != 0){//�I��node�̈ړ�
			long diff;
			int x = e.getX()-20;
			diff =  (long)(x * divx) - llt.get(slc);//diff=�ړ��������̍�
			if(llt.get(slc-1) < llt.get(slc)+diff){//�ړ�����ꏊ���O��note��菬�����Ȃ�Ȃ��̂ł���Έړ�
				for(int i=slc; i<llt.size();i++)
					llt.set(i,llt.get(i)+diff);
				try{
					seq2 = new Sequence(Sequence.PPQ,60000/temposcr.getValue());//�ݒ肵��tempo����
				}catch  (Exception er) {
					er.printStackTrace();
				}
				track2 = seq2.createTrack();
				for(int i = 0;i< llt.size();i++)//�␳����tick�̃��X�g���V�[�P���X��
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
	//�X�N���[���o�[�̊֐�
	public void adjustmentValueChanged(AdjustmentEvent e){
		if(e.getSource() == temposcr)
			tempola.setText(tempost.valueOf(temposcr.getValue()));//�X�N���[���o�[��tempo�\��text�̊֘A�t��
		else if(e.getSource() == tickscr)
			tickla.setText(new String().valueOf(tickscr.getValue()));
		disp();
		repaint();
	}
	//program���܂Ƃ߂邽�߂̊֐�
	public void set(Component com, int x, int y, int sizex, int sizey){//�R���|�[�l���g�̔z�u
		com.setLocation(x,y);
		com.setSize(sizex,sizey);
	}
	public int getnote(int x, int y){//�o�͂��ׂ����̓��肷��֐�
		int n = 0;
		for(int i = 0; i < t.length; i++){
			if(t[i][2] == "0"){//����
				if(x > 20+Integer.parseInt(t[i][1]) && x < 20+Integer.parseInt(t[i][1])+28){
					if(y > 30 && y < 130){
						n = Integer.parseInt(t[i][0]);
						break;
					}
				}
			}
		}
		for(int i = 0; i < t.length; i++){
			if(t[i][2] == "1"){//����
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
	public void addNoteDate(Track track, int note, long tick, long addtick){//note�����g���b�N�ɉ�����
		try{
			ShortMessage mes = new ShortMessage(); //���̊J�n�V���[�g���b�Z�[�W
			mes.setMessage(ShortMessage.NOTE_ON, 0, note, 127);
			track.add(new MidiEvent(mes,tick));
			ShortMessage mesf = new ShortMessage();//���̏I���V���[�g���b�Z�[�W
			mesf.setMessage(ShortMessage.NOTE_OFF, 0, note, 127);
			track.add(new MidiEvent(mesf,tick+addtick));
		}catch(Exception er) {
			er.printStackTrace();
		}
	}
	public void musicStop(){//�V�[�P���T��~
		if(seqer.isRunning() == true)
			seqer.stop();//Sequencer��~
		if(seqer2.isRunning() == true)
			seqer2.stop();//Sequencer��~
	}
}
