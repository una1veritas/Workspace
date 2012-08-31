
import javax.sound.midi.*;

import java.io.*;
import java.util.*;

public class musicPlayer {

	Synthesizer synth=null;
	Sequence seq;
	Sequencer sequencer=null;
	Vector synthInfos;
	
	public void getDriverInfo(){
		synthInfos= new Vector();
		MidiDevice device = null;
		MidiDevice.Info[] infos=MidiSystem.getMidiDeviceInfo();
		for(int i=0;i<infos.length;i++){
			try{
				MidiSystem.getMidiDevice(infos[i]);
			}catch(MidiUnavailableException e){
				e.getCause();
			}
			if(device instanceof Synthesizer){
				synthInfos.add(infos[i]);
			}
		}
	}
	
	public void setMidi(String file) throws FileNotFoundException{
		try {
			sequencer=MidiSystem.getSequencer();
			if(sequencer==null){
				//defaultSynthesizer
				System.out.println("Sequencer");
				System.exit(1);
			}
			sequencer.open();
			File files = new File(file);
			seq=MidiSystem.getSequence(files);
			sequencer.setSequence(seq);
			
		} catch (FileNotFoundException e){
			throw e;
		}
		  catch (MidiUnavailableException e) {
			e.printStackTrace();
		} catch (InvalidMidiDataException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
	}
	
	public void setTimeing(long t){
		sequencer.setMicrosecondPosition(t);
	}
	
	public void setTickTimeing(long t){
		sequencer.setTickPosition(t);
	}
	
	public long getTime(){
		return sequencer.getMicrosecondLength();
	}
	
	public void start(){
		sequencer.start();
	}
	
	public void stop(){
		sequencer.stop();
	}
	
	public void end(){
		sequencer.close();
	}
	//public static void main(String[] args){
		//musicalPlayer mp = new musicalPlayer();
		//mp.setMidi(args[0]);
	//}
}
