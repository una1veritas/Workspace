import java.awt.Color;
import java.awt.Graphics;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.Enumeration;
import java.util.StringTokenizer;
import java.util.Vector;
import javax.imageio.ImageIO;


public class Saveimage {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		String file_name = "994kd.png";
		boolean flag = false;

		Vector pvec = new Vector();
		Vector ovec = new Vector();
		Point p[];
		Point out[];


		String line;
		try{
			BufferedReader reader = new BufferedReader(new FileReader("r994kd.txt"));
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
		
		int xmax;
		int ymax;
		int xmin;
		int ymin;

		if(flag){
			xmax= p[0].x();
			ymax= p[0].y();
			xmin = p[0].x();
			ymin = p[0].y();
			for(int i = 1;i < p.length;i++){
				if(xmin > p[i].x()){
					xmin = p[i].x();
				}
				if(ymin > p[i].y()){
					ymin = p[i].y();
				}
				if(xmax < p[i].x()){
					xmax = p[i].x();
				}
				if(ymax < p[i].y()){
					ymax = p[i].y();
				}
			}
		}else{
			xmax= out[0].x();
			ymax= out[0].y();
			xmin = out[0].x();
			ymin = out[0].y();
			for(int i = 1;i < out.length;i++){
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
		}
		BufferedImage img = new BufferedImage(xmax - xmin +30,ymax - ymin +30, BufferedImage.TYPE_INT_BGR);
		paint(img,flag,p,out, xmin-15,ymin-15);
		try {
			ImageIO.write(img,"png",new File(file_name));
		} catch (IOException ex) {
			ex.printStackTrace();
		}
		

	}
	
	public static void paint(Image img, boolean flag,Point[] p, Point out[], int x, int y) {
		int w = img.getWidth(null); 	//Image‚Ì•
		int h = img.getHeight(null);	//Image‚Ì‚‚³
		double scale = 1.0;
		String readfile = "line_coordinate2.txt";
		int start_x = -1 * x;
		int start_y = -1 * y;
		String line;
		
		
			
		Graphics g = img.getGraphics();
		g.setColor(Color.WHITE);
		g.fillRect(0, 0, w, h);
		if(flag){
			g.setColor(Color.cyan);
			try{
				BufferedReader reader = new BufferedReader(new FileReader(readfile));
				while ((line = reader.readLine()) != null){
					StringTokenizer stk = new StringTokenizer(line,",");
					Integer s = new Integer(stk.nextToken());
					Integer end = new Integer(stk.nextToken());
					if(s.intValue() <= p.length && end.intValue() <= p.length){
						int ox1 = (int)(p[s.intValue()-1].x()*scale)+start_x;
						int oy1 = (int)(p[s.intValue()-1].y()*scale)+start_y;
						int ox2 = (int)(p[end.intValue()-1].x()*scale)+start_x;
						int oy2 = (int)(p[end.intValue()-1].y()*scale)+start_y;
						g.drawLine(ox1,oy1,ox2,oy2);
					}
				}
				reader.close();
				} catch(Exception ex){	
				}
				g.setColor(Color.blue);
				for(int i = 0; i < p.length;i++){
					int px = (int)(p[i].x()*scale)+start_x;
					int py = (int)(p[i].y()*scale)+start_y;
					//g.fillRect(px-2, py-2,4,4);
					g.fillRect(px-5, py-5,10,10);
					//g.drawString(Integer.toString(p[i].num()), px+4, py-4);
				}
		}else{
			g.setColor(Color.gray);
			try{
				BufferedReader reader = new BufferedReader(new FileReader(readfile));
				while ((line = reader.readLine()) != null){
					StringTokenizer stk = new StringTokenizer(line,",");
					Integer s = new Integer(stk.nextToken());
					Integer end = new Integer(stk.nextToken());
					if(s.intValue() <= p.length && end.intValue() <= p.length){
						int ox1 = (int)(out[s.intValue()-1].x()*scale)+start_x;
						int oy1 = (int)(out[s.intValue()-1].y()*scale)+start_y;
						int ox2 = (int)(out[end.intValue()-1].x()*scale)+start_x;
						int oy2 = (int)(out[end.intValue()-1].y()*scale)+start_y;
					g.drawLine(ox1,oy1,ox2,oy2);
					}
				}
				reader.close();
			} catch(Exception ex){
			}
			g.setColor(Color.red);
			for(int i = 0; i < out.length;i++){
				int ox = (int)(out[i].x()*scale)+start_x;
				int oy = (int)(out[i].y()*scale)+start_y;
				g.fillRect(ox-10, oy-10,20,20);
				//g.drawString(Integer.toString(out[i].num()), ox+4, oy+12);
			}
		}
		g.dispose();
	}
}
