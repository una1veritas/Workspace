import java.applet.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Enumeration;
import java.util.StringTokenizer;
import java.util.Vector;

public class Draw extends Applet implements ActionListener{
	private double scale = 1.0;
	Image img;
	String readfile2 = "result.txt";
	String readfile = "line_coordinate2.txt";
	Vector pvec = new Vector();
	Vector ovec = new Vector();
	Vector l = new Vector();
	Point p[];
	Point out[];

	boolean iFlag = false;
	boolean pFlag = true;
	boolean rFlag = true;
	boolean aFlag = false;
	boolean lFlag = false;
	boolean gFlag = false;
	int start_x = 10;
	int start_y = 40;
	
	public void init(){
		img = getImage(getDocumentBase(), "226_kk.jpg");
		resize(1200,800);
		
		Button img_b = new Button("img");
		Button pattern_b= new Button("pattern");
		Button rei_b= new Button("レイアウト");	
		Button arrow_b= new Button("Arrow");
		Button line_b= new Button("line");
		Button grid_b= new Button("grid");
		Button scaleUp= new Button("Up");
		Button scaleDown= new Button("Down");

		String line;
		try{
			BufferedReader reader = new BufferedReader(new FileReader(readfile2));
			int number = 0;
			while ((line = reader.readLine()) != null){
				if(line.startsWith("#")){
					System.out.println("cut");
				}else{
					StringTokenizer stk = new StringTokenizer(line,",");
					Integer dx = new Integer(stk.nextToken());
					Integer dy = new Integer(stk.nextToken());
					Integer odx = new Integer(stk.nextToken());
					Integer ody = new Integer(stk.nextToken());
					pvec.addElement(new Point(dx.intValue(),dy.intValue(),number));
					ovec.addElement(new Point(odx.intValue(),ody.intValue(),number));
					number++;
				}
			}
			reader.close();
		} catch(Exception ex){
		
		}
		p = new Point[pvec.size()];
		Enumeration e = pvec.elements();
		for(int i = 0; i < p.length;i++){
			p[i] = new Point((Point)e.nextElement());
		}

		
		
		out = new Point[ovec.size()];
		e = ovec.elements();
		for(int i = 0; i < out.length;i++){
			out[i] = new Point((Point)e.nextElement());
		}
	     this.add(img_b);
	     img_b.addActionListener(this);
	     this.add(pattern_b);
	     pattern_b.addActionListener(this);
	     this.add(rei_b);
	     rei_b.addActionListener(this);
	     this.add(arrow_b);
	     arrow_b.addActionListener(this);
	     this.add(line_b);
	     line_b.addActionListener(this);
	     this.add(grid_b);
	     grid_b.addActionListener(this);
	     this.add(scaleUp);
	     scaleUp.addActionListener(this);
	     this.add(scaleDown);
	     scaleDown.addActionListener(this);
	     
	     int xmin = out[0].x();
	     int ymin = out[0].y();
	     int xmax = out[0].x();
	     int ymax = out[0].y();
	     for(int i =1;i < out.length;i++){
	    	 if(xmin > out[i].x()){
	    		 xmin = out[i].x();
	    	 }
	    	 if(ymin > out[i].y()){
	    		 ymin = out[i].y();
	    	 }
	    	 if(xmax < out[i].x()){
	    		 xmax = out[i].x();
	    	 }
	    	 if(ymax < out[i].y()){
	    		 ymax = out[i].y();
	    	 }
	     }
	     double a = (double)out.length/(((xmax-xmin)/60+1)*((ymax-ymin)/60+1));
	     System.out.println(a);
	}
	
	public void paint(Graphics g){
		g.drawString("x"+Double.toString(scale), 800, 20);
		if(iFlag){
			g.drawImage(img, start_x, start_y, this);
		}
		
		if(gFlag){
			g.setColor(Color.DARK_GRAY);
			int grid = (int)(60*scale);
			for(int i = start_x;i <= 1000 + start_x;i += grid){
				g.drawLine(i,start_y,i,1000+start_y);
			}
			for(int i = start_y;i <= 1000 + start_y;i += grid){
				g.drawLine(start_x,i,1000+start_x,i);
			}
		}
		

		if(aFlag){
			ExGraphics eg= new ExGraphics(g);
			eg.setColor(Color.ORANGE);
			for(int i = 0; i < out.length;i++){
				int px = (int)(p[i].x()*scale)+start_x;
				int py = (int)(p[i].y()*scale)+start_y;
				int ox = (int)(out[i].x()*scale)+start_x;
				int oy = (int)(out[i].y()*scale)+start_y;
				eg.drawWarrow(px, py, ox, oy,10,3);
			}
		}
		if(pFlag){
			if(lFlag){
				g.setColor(Color.cyan);
				try{
					BufferedReader reader = new BufferedReader(new FileReader(readfile));
					String line;
					while ((line = reader.readLine()) != null){
						StringTokenizer stk = new StringTokenizer(line,",");
						Integer s = new Integer(stk.nextToken());
						Integer e = new Integer(stk.nextToken());
						if(s.intValue() <= p.length && e.intValue() <= p.length){
							int ox1 = (int)(p[s.intValue()-1].x()*scale)+start_x;
							int oy1 = (int)(p[s.intValue()-1].y()*scale)+start_y;
							int ox2 = (int)(p[e.intValue()-1].x()*scale)+start_x;
							int oy2 = (int)(p[e.intValue()-1].y()*scale)+start_y;
							g.drawLine(ox1,oy1,ox2,oy2);
						}
					}
					reader.close();
				} catch(Exception ex){
				
				}
			}
			g.setColor(Color.blue);
			for(int i = 0; i < p.length;i++){
				int px = (int)(p[i].x()*scale)+start_x;
				int py = (int)(p[i].y()*scale)+start_y;
				g.fillRect(px-2, py-2,4,4);
				g.drawString(Integer.toString(p[i].num()), px+4, py-4);
			}
		}
		if(rFlag){	
			if(lFlag){
				g.setColor(Color.gray);
				try{
					BufferedReader reader = new BufferedReader(new FileReader(readfile));
					String line;
					while ((line = reader.readLine()) != null){
						StringTokenizer stk = new StringTokenizer(line,",");
						Integer s = new Integer(stk.nextToken());
						Integer e = new Integer(stk.nextToken());
						if(s.intValue() <= p.length && e.intValue() <= p.length){
							int ox1 = (int)(out[s.intValue()-1].x()*scale)+start_x;
							int oy1 = (int)(out[s.intValue()-1].y()*scale)+start_y;
							int ox2 = (int)(out[e.intValue()-1].x()*scale)+start_x;
							int oy2 = (int)(out[e.intValue()-1].y()*scale)+start_y;
							g.drawLine(ox1,oy1,ox2,oy2);
						}
					}
					reader.close();
				} catch(Exception ex){
				
				}
			}
			g.setColor(Color.red);
			for(int i = 0; i < out.length;i++){
				int ox = (int)(out[i].x()*scale)+start_x;
				int oy = (int)(out[i].y()*scale)+start_y;
				g.fillRect(ox-3, oy-3,6,6);
				g.drawString(Integer.toString(out[i].num()), ox+4, oy+12);
		}
		}
		
	}
	
	public void actionPerformed(ActionEvent e){
		if(e.getActionCommand().equals("img")){
			if(iFlag == true)
				iFlag = false;
			else
				iFlag = true;
		}
		if(e.getActionCommand().equals("pattern")){
			if(pFlag == true)
				pFlag = false;
			else
				pFlag = true;
		}
		if(e.getActionCommand().equals("レイアウト")){
			if(rFlag == true)
				rFlag = false;
			else
				rFlag = true;
		}
		if(e.getActionCommand().equals("line")){
			if(lFlag == true)
				lFlag = false;
			else
				lFlag = true;
		}
		if(e.getActionCommand().equals("grid")){
			if(gFlag == true)
				gFlag = false;
			else
				gFlag = true;
		}
		if(e.getActionCommand().equals("Arrow")){
			if(aFlag == true)
				aFlag = false;
			else
				aFlag = true;
		}
		if(e.getActionCommand().equals("Up")){
			scale *= 2;
		}
		if(e.getActionCommand().equals("Down")){
			scale /= 2;
		}
		repaint();
	}
}
