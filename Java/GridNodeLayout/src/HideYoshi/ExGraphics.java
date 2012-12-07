import java.awt.*;
import java.awt.image.ImageObserver;
import java.text.AttributedCharacterIterator;

/**
 *　Graphics クラスを拡張した ExGraphics クラス<BR>
 *　　福野さんのNullGraphics クラスを参考にさせていただきました。<BR>
 *
 *　Graphics クラスを継承し、以下の６つのメソッドを追加します。<BR>
 *　　void drawCircle(int x,int y,int r)　中心を指定して円を描くメソッド<BR>
 *　　　　　　　　　中心(x,y)、半径 r の円を描きます。<BR>
 *　　void fillCircle(int x,int y,int r)  中心を指定して円を塗るメソッド<BR>
 *　　　　　　　　　中心(x,y)、半径 r の円を描き中を塗りつぶします。<BR>
 *　　void drawArrow(int x0,int y0,int x1,int y1,int l)  矢印を描画するメソッド<BR>
 *　　　　　　　　　始点(x0,y0)から終点(x1,y1)へ矢印を描きます。<BR>
 *　　　　　　　　　矢の長さを l で与えます。<BR>
 *　　void drawWline(int x0,int y0,int x1,int y1,int w) 太い直線を描画するメソッド<BR>
 *　　　　　　　　　始点(x0,y0)から終点(x1,y1)へ太さ w の直線を描きます。<BR>
 *　　void drawWarrow(int x0,int y0,int x1,int y1,int l,int w)  太い矢印を描画するメソッド<BR>
 *　　　　　　　　　始点(x0,y0)から終点(x1,y1)へ矢印を描きます。<BR>
 *　　　　　　　　　矢の長さを l で与えます。太さを w で与えます。<BR>
 *　　void drawWcircle(int x, int y, int r, int w)　太さを指定して円を描くメソッド<BR>
 *　　　　　　　　　中心(x,y)　半径 r の円を w の線で描きます。<BR>
 */
public class ExGraphics extends Graphics {
    Graphics g;
    
    public ExGraphics(Graphics g) {
      this.g = g;
    }
    public Graphics create() {
      return new ExGraphics(g.create());
    }
    public Graphics create(int x, int y, int width, int height) {
      return new ExGraphics(g.create(x, y, width, height));
    }
    public void translate(int x, int y){
      g.translate(x, y);
    }
    public Color getColor(){
      return g.getColor();
    }
    public void setColor(Color c){
      g.setColor(c);
    }
    public void setPaintMode(){
      g.setPaintMode();
    }
    public void setXORMode(Color c1){
      g.setXORMode(c1);
    }
    public Font getFont(){
      return g.getFont();
    }
    public void setFont(Font font){
      g.setFont(font);
    }
    public FontMetrics getFontMetrics() {
      return g.getFontMetrics();
    }
    public FontMetrics getFontMetrics(Font f){
      return g.getFontMetrics(f);
    }
    public Rectangle getClipBounds(){
      return g.getClipBounds();
    }
    public void clipRect(int x, int y, int width, int height){
      g.clipRect( x, y, width, height);
    }
    public void setClip(int x, int y, int width, int height){
      g.setClip( x, y, width, height);
    }
    public Shape getClip(){
      return g.getClip();
    }
    public void setClip(Shape clip){
      g.setClip(clip);
    }
    public void copyArea(int x, int y, int width, int height,
				  int dx, int dy){
      g.copyArea( x, y, width, height, dx, dy);
    }
    public void drawLine(int x1, int y1, int x2, int y2){
      g.drawLine( x1, y1, x2, y2);
    }
    public void fillRect(int x, int y, int width, int height){
      g.fillRect( x, y, width, height);
    }
    public void drawRect(int x, int y, int width, int height){
      g.drawRect( x, y, width, height);
    }
    public void clearRect(int x, int y, int width, int height){
      g.clearRect( x, y, width, height);
    }
    public void drawRoundRect(int x, int y, int width, int height,
				       int arcWidth, int arcHeight){
      g.drawRoundRect( x, y, width, height, arcWidth, arcHeight);
    }
    public void fillRoundRect(int x, int y, int width, int height,
				       int arcWidth, int arcHeight){
      g.fillRoundRect( x, y, width, height, arcWidth, arcHeight);
    }
    public void draw3DRect(int x, int y, int width, int height,
			   boolean raised){
      g.draw3DRect( x, y, width, height, raised);
    }
    public void fill3DRect(int x, int y, int width, int height,
			   boolean raised){
      g.fill3DRect( x, y, width, height, raised);
    }
    public void drawOval(int x, int y, int width, int height){
      g.drawOval( x, y, width, height);
    }
    public void fillOval(int x, int y, int width, int height){
      g.fillOval( x, y, width, height);    }
    public void drawArc(int x, int y, int width, int height,
				 int startAngle, int arcAngle){
      g.drawArc( x, y, width, height, startAngle, arcAngle);
    }
    public void fillArc(int x, int y, int width, int height,
				 int startAngle, int arcAngle){
      g.fillArc( x, y, width, height, startAngle, arcAngle);
    }
    public void drawPolyline(int xPoints[], int yPoints[],
				      int nPoints){
      g.drawPolyline( xPoints, yPoints, nPoints);
    }
    public void drawPolygon(int xPoints[], int yPoints[],
				     int nPoints){
      g.drawPolygon( xPoints, yPoints, nPoints);
    }
    public void drawPolygon(Polygon p){
      g.drawPolygon(p);
    }
    public void fillPolygon(int xPoints[], int yPoints[],
				     int nPoints){
      g.fillPolygon( xPoints, yPoints, nPoints);
    }
    public void fillPolygon(Polygon p){
      g.fillPolygon(p);
    }
    public void drawString(String str, int x, int y){
      g.drawString( str, x, y);
    }
    public void drawChars(char data[], int offset, int length, int x, int y){
      g.drawChars( data, offset, length, x, y);
    }
    public void drawBytes(byte data[], int offset, int length, int x, int y){
      g.drawBytes( data, offset, length, x, y);
    }
    public boolean drawImage(Image img, int x, int y, 
				      ImageObserver observer){
      return g.drawImage( img, x, y, observer);
    }
    public boolean drawImage(Image img, int x, int y,
				      int width, int height, 
				      ImageObserver observer){
      return g.drawImage( img, x, y, width, height, observer);
    }
    public boolean drawImage(Image img, int x, int y, 
				      Color bgcolor,
				      ImageObserver observer){
      return g.drawImage( img, x, y, bgcolor, observer);
    }
    public boolean drawImage(Image img, int x, int y,
				      int width, int height, 
				      Color bgcolor,
				      ImageObserver observer){
      return g.drawImage( img, x, y, width, height, bgcolor, observer);
    }
    public boolean drawImage(Image img,
				      int dx1, int dy1, int dx2, int dy2,
				      int sx1, int sy1, int sx2, int sy2,
				      ImageObserver observer){
      return g.drawImage(img,dx1, dy1, dx2, dy2, sx1, sy1, sx2, sy2, observer);
    }
    public boolean drawImage(Image img,
				      int dx1, int dy1, int dx2, int dy2,
				      int sx1, int sy1, int sx2, int sy2,
				      Color bgcolor,
				      ImageObserver observer){
      return g.drawImage(img,dx1,dy1,dx2,dy2,sx1,sy1,sx2,sy2,bgcolor,observer);
    }
    public void dispose(){
      g.dispose();
    }
    public void finalize(){
      g.finalize();
    }
    public String toString(){
      return g.toString();
    }
//    public Rectangle getClipRect(){
//      return g.getClipRect();
//    }

/**
 *　以下、拡張部分
 */
 /**
  *　中心を指定して円を描くメソッド<BR>
  *　　void drawCircle(int x,int y,int r)<BR>
  *　　中心(x,y)、半径 r の円を描きます。<BR>
  */
  public void drawCircle(int x,int y,int r){
    g.drawOval(x-r,y-r,r*2,r*2);
  }
  
 /**
  *　中心を指定して円を塗るメソッド<BR>
  *　　void fillCircle(int x,int y,int r)<BR>
  *　　中心(x,y)、半径 r の円を描き中を塗りつぶします。<BR>
  */
  public void fillCircle(int x,int y,int r){
    g.fillOval(x-r,y-r,r*2,r*2);
  }
  
 /**
  *　矢印を描画するメソッド<BR>
  *　　void drawArrow(int x0,int y0,int x1,int y1,int l)<BR>
  *　　始点(x0,y0)から終点(x1,y1)へ矢印を描きます。<BR>
  *　　矢の長さを l で与えます。<BU>
  */
  public void drawArrow(int x0,int y0,int x1,int y1,int l){
    double theta;
    int x,y;
    double dt = Math.PI / 6.0;
    theta = Math.atan2((double)(y1-y0),(double)(x1-x0));
    g.drawLine(x0,y0,x1,y1);
    x = x1-(int)(l*Math.cos(theta-dt));
    y = y1-(int)(l*Math.sin(theta-dt));
    g.drawLine(x1,y1,x,y);
    x = x1-(int)(l*Math.cos(theta+dt));
    y = y1-(int)(l*Math.sin(theta+dt));
    g.drawLine(x1,y1,x,y);
  }

 /**
  *　太い直線を描画するメソッド<BR>
  *　　void drawWline(int x0,int y0,int x1,int y1,int w)<BR>
  *　　始点(x0,y0)から終点(x1,y1)へ太さ w の直線を描きます。<BR>
  */
  public void drawWline(int x0,int y0,int x1,int y1,int w){
    int dx,dy,nxi,nyi;
    double d,nx,ny;
    int i;
    int[] x = new int[4];
    int[] y = new int[4];
    
    dx = x1-x0;
    dy = y1-y0;
    d = Math.sqrt(dx*dx+dy*dy);
    nx = (double)dy / d;
    ny = -(double)dx / d;
    nxi = (int)(Math.rint(nx*w/2.0));
    nyi = (int)(Math.rint(ny*w/2.0));
    x[0] = x0-nxi;
    y[0] = y0-nyi;
    x[1] = x1-nxi;
    y[1] = y1-nyi;
    x[2] = x1+nxi;
    y[2] = y1+nyi;
    x[3] = x0+nxi;
    y[3] = y0+nyi;
    g.fillPolygon(x, y, 4);
  }

 /**
  *　太い矢印を描画するメソッド<BR>
  *　　void drawWarrow(int x0,int y0,int x1,int y1,int l,int w)<BR>
  *　　始点(x0,y0)から終点(x1,y1)へ矢印を描きます。<BR>
  *　　矢の長さを l で与えます。太さを w で与えます。<BR>
  */
  public void drawWarrow(int x0,int y0,int x1,int y1,int l,int w){
    double theta;
    int x,y;
    double dt = Math.PI / 6.0;
    theta = Math.atan2((double)(y1-y0),(double)(x1-x0));
    drawWline(x0,y0,x1,y1,w);
    x = x1-(int)(l*Math.cos(theta-dt));
    y = y1-(int)(l*Math.sin(theta-dt));
    drawWline(x1,y1,x,y,w);
    x = x1-(int)(l*Math.cos(theta+dt));
    y = y1-(int)(l*Math.sin(theta+dt));
    drawWline(x1,y1,x,y,w);

  }

 /**
  *　太さを指定して円を描くメソッド<BR>
  *　　void drawWcircle(int x, int y, int r, int w)<BR>
  *　　中心(x,y)　半径 r の円を w の線で描きます。<BR>
  */
  void drawWcircle(int x, int y, int r, int w){
    int i;
    
    for (i=r-(int)(w/2);i<=r+(int)(w/2);i++){
      drawCircle(x,y,i);
    }
  }
public void drawString(AttributedCharacterIterator iterator, int x, int y) {
	// TODO 自動生成されたメソッド・スタブ
	
}

}

