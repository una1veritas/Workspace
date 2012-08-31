

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
		// TODO �����������ꂽ�R���X�g���N�^�[�E�X�^�u
	}
	//�p�^�[���ǂݍ���//���͂̌`������ŕς��
	//���̂Ƃ����
	/*
	 * [�f���^�^�C�� ����]�����p�X�y�[�X�ŋ�؂��ĕ����A�Ȃ����`�����Ă���Ɖ��肷��
	 * �͂��߂̈�����̓f���^�^�C�������łɓ��͂���Ă�����̂Ƃ���i
	 */
	public SMFScore(String pat){
		format=0;
		tracks=1;
		division=480;
		int channel=1;
		trackLength=new int[1];
		Vector<MusicalNote> score=new Vector<MusicalNote>();
		StringTokenizer st=new StringTokenizer(pat);

		//�����̏�񂪂Ȃ��ꍇ���̃f���^�^�C���͔j������
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
	
	
	//�ϒ��\������l�����o��
	private static int parseVarLenInt(InputStream strm) throws IOException {
		int oneByte, value = 0; // for storing unsigned byte
		//InputStream.read()--�X�g���[���̏I���ɒB���ēǂݍ��ރf�[�^���Ȃ��ꍇ�� -1 ��Ԃ��܂�
		while ( (oneByte = strm.read()) != -1 ) {
			value = value << 7;
			value += oneByte & 0x7f;
			/*
			oneByte ? ? ? ? ? ? ? ?
			0x7f    0 1 1 1 1 1 1 1 
			�ebit��AND����������̂�value�ɓ����
			oneByte��1bit�ڂ̓X�e�[�^�X�r�b�g
			 */
			//SMF�̃f�[�^�\���ŃX�e�[�^�X�r�b�g���Z�b�g����Ă�����㑱�o�C�g�����邱�Ƃ������̂�
			//�㑱�o�C�g�������Ȃ�܂ŉ񂷁B
			if ( (oneByte & 0x80) == 0 )
				break;			
		}
		//�t�@�C���̏I���
		if ( oneByte == -1 )
			return -1;
		return value;
	}
	
	//�Œ蒷���l�\������l�����o�� �p�r�̓��^�C�x���g�̂݁H
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
	
	//�w�b�_�`�����N���̃f�[�^������
	public boolean headChunk(InputStream strm) throws IOException{

		byte[] buf ={0,0,0,0};
		strm.read(buf);
		if(buf[0]=='M'&&buf[1]=='T'&&buf[2]=='h'&buf[3]=='d'){
			strm.skip(4);//�f�[�^�T�C�Y���X�L�b�v
			format=strm.read() << 8; //���byte�����č���8bit�V�t�g
			format+=strm.read();     //����byte�𑫂�

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

	//�g���b�N�`�����N���̃f�[�^������
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
		//�����j���O�X�e�[�^�X�ł���t���O	 true�E�E�����j���O�X�e�[�^�X��
		boolean runFlag=false;
		//�������̃m�[�g
		//���������`���l���ԍ��A���������m�[�g�ԍ��A���e��Score���̈ʒu
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

			//NoteOn�m�[�g�̏�����
			for(int c=0;c<channelSize;c++){
				for(int n=0;n<noteSize;n++){
					noteOn[c][n]=null;
				}
			}
			while(true){//�e�����𔲂����Ƃ�����1byte�͕K���f���^�^�C��
				//�f���^�^�C���̎擾
				deltaTime=parseVarLenInt(strm);
				//�I������
				if(deltaTime==-1){
					break;
				}
				
				deltaTotal+=deltaTime;
				oneByte=strm.read();
				runFlag=true;
				if((oneByte & 0x80)!=0){
					//�V�X�e���o�C�g����
//					���4bit�Ɖ���4bit�ɕ�����
					high4bits=((oneByte & 0xf0) >> 4);
					low4bits=oneByte & 0x0f;
					if(oneByte ==0xf0){
						//�V�X�e���G�N�X�N���[�V�u
						//System.out.println("systemExclusiveMessage");
						while(oneByte!=0xf7){
							oneByte=strm.read();
						}
						continue;
					}
					else if(oneByte==0xf7){
						//�G���h�I�u�G�N�X�N���[�V�u
						//System.out.println("EOX");
						len=parseVarLenInt(strm);
						//�V�X�e�����A���^�C�����b�Z�[�W�A�\���O�|�W�V�����|�C���^�[�A
						//�\���O�Z���N�g�AMIDI�^�C���R�[�h����������ꍇ�͂����ɋL�q
						for(int n=0;n<len;n++){
							strm.read();
						}
						continue;
					}
					else if(oneByte==0xff){
						//���^�C�x���g
						//�C�x���g�^�C�v�̎擾
						oneByte=strm.read();
						if(oneByte==0x2f){
							//End_of_Track
//							16�`���l���̉�����ɂ܂Ƃ߂�
							for(int n=0;n<channelSize;n++){
								if(n==9||channelScore[n].size()==0)//�e���|�͖�������
									continue;
								trackLength[i]=Math.max(trackLength[i],((MusicalNote)channelScore[n].get(channelScore[n].size()-1)).noteOn);
								score.addAll(score.size(),channelScore[n]);
								score.add(new MusicalNote(-1,-1,-1,-1,-1));
								channelScore[n].clear();
							}

							//�C�x���g�T�C�Y��ǂݔ�΂�
							strm.read();
							break;
						}
						else if(oneByte==0x51){//�Z�b�g�e���|
							int len2=parseVarLenInt(strm);//�C�x���g�T�C�Y3��ǂݔ�΂�
							int tttttt=parseVarMetaInt(strm, len2);
							tempo=(int)(1000000*60.0/tttttt);
							//tttttt�͎l���������}�C�N���b�P�ʂŕ\���Ă���̂�
							//BPM�ɕϊ�����B
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
//							�C�x���g�T�C�Y�̎擾
							len=parseVarLenInt(strm);
							for(int j=0;j<len;j++){
								//���^�C�x���g�𖳎�����
								strm.read();
							}
							continue;
						}
					}
					//�f�[�^�o�C�g�擾
					//�e�������ƂɎ擾���鎖�ɂ���(�ύX�ɋ������邽��)
					//oneByte=strm.read();
					runFlag=false;
				}
				/*
//				���4bit�Ɖ���4bit�ɕ�����
				high4bits=((oneByte & 0xf0) >> 4);
				low4bits=oneByte & 0x0f;
				*/
				//MIDI�C�x���g����
				switch(high4bits){
				case 0x08://noteOff low4bits�̓`���l���ԍ�
					//NoteNumber
				/*�����j���O�X�e�[�^�X�̏ꍇ���łɏ��oneByte�͓ǂݍ����
				����̂ł����ł͓ǂݍ��܂Ȃ��B�ȉ��̃X�e�[�^�X�̏ꍇ�����l
				�C�x���g���ƂɃf�[�^��ǂݍ��܂��ɃX�e�[�^�X�o�C�g�̂Ƃ��㑱
				�P�o�C�g��oneByte�ɓǂݍ��߂�runFlag�͎g��Ȃ��čς�
				*/
					if(!runFlag)
						oneByte=strm.read();
					//Velocity
					strm.read();
					//�m�[�g�I�t�̏���score�ɕt������
					if(oneByte>noteSize){
						System.out.println(low4bits+" : "+oneByte);
						
					}
					if(noteOn[low4bits][oneByte]!=null){
						noteOn[low4bits][oneByte].setDuration(deltaTotal);
						noteOn[low4bits][oneByte]=null;
					}
					break;
				case 0x09://noteOn low4bits�̓`���l���ԍ�
					noteNum++;
//					NoteNumber
					if(!runFlag)
						oneByte=strm.read();
					//Velocity
					velocity=strm.read();
					if(velocity==0x00)
						//noteOf�̏���������
						try{
						if(noteOn[low4bits][oneByte]!=null){
							noteOn[low4bits][oneByte].setDuration(deltaTotal);
							noteOn[low4bits][oneByte]=null;
							break;
						}else{//noteOff�Ƃ��Ă��g���Ă��Ȃ���?
							break;
						}
						}catch(ArrayIndexOutOfBoundsException e){
							throw e;
							//for(int h=0;;)
								//System.out.println(Integer.toHexString(strm.read()));
						}
					//ch9�̓��Y��
					if(low4bits==9){
						continue;
					}
					
					if(noteOn[low4bits][oneByte]!=null){//�I����������Ă��Ȃ������炳�ꂽ�Ƃ�
														//�ォ��������𖳎�����(�b��)
						break;
						//�������Ȃ��ꍇduration��t������
						//noteOn[low4bits][oneByte].setDuration(deltaTotal);
					}

					channelScore[low4bits].add(new MusicalNote(low4bits+1,deltaTotal,oneByte,velocity,tempo));
					noteOn[low4bits][oneByte]=(MusicalNote)channelScore[low4bits].lastElement();
					
					break;
				case 0x0a://�|���t�H�j�b�N�L�[�v���b�V���[
					if(!runFlag)
						strm.read();//NoteNumber
					strm.read();//Pressure
					break;
				case 0x0b://�R���g���[���`�F���W
					if(!runFlag)
						strm.read();
					//Control�@�`���l�����[�h���b�Z�[�W����������Ƃ��͂����ɋL�q
					strm.read();//Value
					break;
				case 0x0c://�v���O�����`�F���W
						//�m�[�g�I�t�A�I������������?
					if(!runFlag)
						strm.read();
					break;
				case 0x0d://�`���l���v���b�V���[
					if(!runFlag)
						strm.read();//Pressure
					break;
				case 0x0e://�s�b�`�x���h�`�F���W
					if(!runFlag)
						strm.read();//LSB
					strm.read();//MSB
					break;
				}
			}
			//����ςȂ��ŏI��������,�I���������m�[�g�I�t�̎����Ƃ���
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
	//�Ȓ��ōŌ�ɂȂ������̎��Ԃ�Ԃ�
	public int getLongest(){
		int tmp=-1;
		for(int i=0;i<trackLength.length;i++){
			tmp=Math.max(tmp,trackLength[i]);
		}
		return tmp;
	}
	//�g���b�Ni�łōŌ�ɖ������̎��Ԃ�Ԃ�
	public int getLastTime(int i){
		return trackLength[i];
	}
//	�e�L�X�g��i�Ԗڂ̉���Ԃ�
	MusicalNote noteAt(int i) {
		return score[i];
		//return score.elementAt(i);
	}
	//�e�L�X�g�̃T�C�Y��Ԃ�
	int size() {
		return score.length;
		//return score.size();
	}
//	����\��Ԃ�
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
			if(fileUpdata)//�t�@�C���Ƀf�[�^��ǉ�
				bw=new BufferedWriter(new FileWriter(filePath,true));
			else{//�t�@�C���V�K�쐬
				bw=new BufferedWriter(new FileWriter(filePath,false));
				fileUpdata=true;
			}	
		} catch (FileNotFoundException e) {
			// TODO �����������ꂽ catch �u���b�N
			e.printStackTrace();
		} catch (IOException e) {
			// TODO �����������ꂽ catch �u���b�N
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
