/*
    このｸﾗｽは java.awt.applet.Applet ｸﾗｽを拡張したものです.
 */

import java.awt.*;
import java.applet.*;
import java.util.*;

public class NumberPlaceSolver extends Applet
{    
	public void init()
	{
		// symantec.itools.net.RelativeURL か symantec.itools.awt.util.StatusScroller
		// を使用しない場合は、以下の行を削除してください
		//symantec.itools.lang.Context.setApplet(this);
	
		// 以下の，ｺﾝﾎﾟｰﾈﾝﾄの生成と初期化を行うｺｰﾄﾞは，VisucalCafeに
		// よって自動生成されました．このｺｰﾄﾞを変更する場合は，
		// VisualCafeが生成するｺｰﾄﾞに沿ったｺｰﾃﾞｨﾝｸﾞ形式で記述してください．
		// さもないと，VisualCafeはﾌｫｰﾑﾃﾞｻﾞｲﾅ側に変更を反映させること
		// ができないかもしれません．
		//{{INIT_CONTROLS
		setLayout(null);
		setSize(370,280);
		numsetButton.setActionCommand("button");
		numsetButton.setLabel("数を固定");
		add(numsetButton);
		numsetButton.setBackground(java.awt.Color.lightGray);
		numsetButton.setBounds(264,24,84,24);
		releaseButton.setActionCommand("button");
		releaseButton.setLabel("固定解除");
		add(releaseButton);
		releaseButton.setBackground(java.awt.Color.lightGray);
		releaseButton.setBounds(264,60,84,24);
		solveButton.setActionCommand("button");
		solveButton.setLabel("自動解答");
		add(solveButton);
		solveButton.setBackground(java.awt.Color.lightGray);
		solveButton.setBounds(264,144,84,24);
		clearButton.setActionCommand("button");
		clearButton.setLabel("記入を消去");
		add(clearButton);
		clearButton.setBackground(java.awt.Color.lightGray);
		clearButton.setBounds(264,216,84,24);
		boxesPanel.setLayout(null);
		add(boxesPanel);
		boxesPanel.setBackground(java.awt.Color.white);
		boxesPanel.setBounds(24,24,222,222);
		//}}
		for (int r = 0; r < 9; r++) {
		    for (int c = 0; c < 9; c++) {
		        tmpfield = new TextField();
        		//tmpfield.setFont(new Font("Dialog", Font.PLAIN, 18));		        
        		tmpfield.setFont(new Font("SansSerif", Font.BOLD, 18));		        
		        tmpfield.setBounds(24*c+(c/3*3), 24*r+(r/3*3), 24, 24);
		        table[r][c] = tmpfield;
		        boxesPanel.add(tmpfield);
		    }
		}
	
		//{{REGISTER_LISTENERS
		SymAction lSymAction = new SymAction();
		numsetButton.addActionListener(lSymAction);
		releaseButton.addActionListener(lSymAction);
		solveButton.addActionListener(lSymAction);
		clearButton.addActionListener(lSymAction);
		//}}
		SymFocus aSymFocus = new SymFocus();
		for (int r = 0; r < 9; r++) 
		    for (int c = 0; c < 9; c++)
		        table[r][c].addFocusListener(aSymFocus);
	}
		
	//{{DECLARE_CONTROLS
	java.awt.Button numsetButton = new java.awt.Button();
	java.awt.Button releaseButton = new java.awt.Button();
	java.awt.Button solveButton = new java.awt.Button();
	java.awt.Button clearButton = new java.awt.Button();
	java.awt.Panel boxesPanel = new java.awt.Panel();
	//}}
	
	TextField[][] table = new TextField[9][9];
	TextField tmpfield;
	

	class SymAction implements java.awt.event.ActionListener
	{
		public void actionPerformed(java.awt.event.ActionEvent event)
		{
			Object object = event.getSource();
			if (object == numsetButton)
				numset_Action(event);
			else if (object == releaseButton)
				release_Action(event);
			else if (object == solveButton)
				solve_Action(event);
			else if (object == clearButton)
				clear_Action(event);
		}
	}

	void numset_Action(java.awt.event.ActionEvent event)
	{
		// to do: ｺｰﾄﾞをここに記述します.
		int content;
		for (int r = 0; r < 9; r++) {
		    for (int c = 0; c < 9; c++) {
		        content = 0;
		        try {
		            content = (new Integer(table[r][c].getText())).intValue();
		        } catch (NumberFormatException exc) {
		            content = 0;
		        }
		        if (1 <= content && content <= 9) {
		            table[r][c].setEnabled(false);
		        } else {
		            table[r][c].setText("");
		        }
		    }
		}
	}
	
	void release_Action(java.awt.event.ActionEvent event)
	{
		// to do: ｺｰﾄﾞをここに記述します.
		for (int r = 0; r < 9; r++)
		    for (int c = 0; c < 9; c++)
	            table[r][c].setEnabled(true);
	}
	
	void clear_Action(java.awt.event.ActionEvent event)
	{
		// to do: ｺｰﾄﾞをここに記述します.
		for (int r = 0; r < 9; r++)
		    for (int c = 0; c < 9; c++)
		        if (table[r][c].isEnabled()) {
		            table[r][c].setText("");
		        }
	}
	
	void solve_Action(java.awt.event.ActionEvent event) {
	    solveButton.setEnabled(false);
	    solveButton.setLabel("考え中...");
	    solve();
	    solveButton.setLabel("解答探索");
	    solveButton.setEnabled(true);
    }
	
	protected class ByteSet {
	    byte[] bytes = new byte[9];
	    
	    protected ByteSet() {
	        super();
	        emptize();
	    }
	    	    
	    protected void emptize() {
	        for (int i = 0; i < 9; i++) {
	            bytes[i] = 0;
	        }
	    }
	    
	    protected void fullize() {
	        for (int i = 0; i < 9; i++) {
	            bytes[i] = 1;
	        }
	    }
	    
	    protected void add(int i) {
	        if (1 <= i && i <= 9)
	            bytes[i-1] = 1;
	    }
	    
	    protected void remove(int i) {
	        if (1 <= i && i <= 9)
	            bytes[i-1] = 0;
	    }
	    
	    protected void remove(ByteSet set) {
	        for (int i = 1; i <= 9; i++) {
	            if (set.contains(i))
	                this.remove(i);
	        }
	    }
	    
	    protected boolean contains(int i) {
	        if (1 <= i && i <= 9)
	            return (bytes[i-1] == 1);
	        else
	            return false;
	    }
	    
	    protected boolean isSingletonSet() {
	        int cnt = 0;
	        for (int i = 1; i <= 9; i++) {
	            if (contains(i)) cnt++;
	        }
	        return (cnt == 1);
	    }
	    
	    protected int smallestValue() {
	        for (int i = 1; i <= 9; i++) {
	            if (contains(i)) return i;
	        }
	        return 0;
	    }
	    
	}
		
	void solve() {
	    ByteSet[][] numset = new ByteSet[9][9];
	    ByteSet tmp = new ByteSet();
	    boolean updated;
	    
	    // initialize the set of possible numbers
	    for (int r = 0; r < 9; r++) {
	        for (int c = 0; c < 9; c++) {
                int val;
                try {
                    val = (new Integer(table[r][c].getText())).intValue();
                } catch (NumberFormatException ex) {
                    val = 0;
                }
                numset[r][c] = new ByteSet();
                if (val == 0) {
                    numset[r][c].fullize();  // set bits from 1 to 9
                } else {
                    numset[r][c].emptize();
                    numset[r][c].add(val);  // indicate as a determined cell.
                }
	        }
	    }
	    
	    updated = true;
    	while (updated) {
    	    updated = false;
    	    
    	    // eliminate impossible numbers by determined box
    	    for (int r = 0; r < 9; r++) {
    	        for (int c = 0; c < 9; c++) {
    	            if (! numset[r][c].isSingletonSet()) 
    	                continue;
    	            // numset[r][c] is singleton set;
    	            for (int dr = 0; dr < 9; dr++) {
                        for (int dc = 0; dc < 9; dc++) {
                            if (numset[dr][dc].isSingletonSet() || (dr == r && dc == c)) 
                                continue; 
                            if (dr == r || dc == c || ((r/3 == dr/3) && (c/3 == dc/3))) {
                                numset[dr][dc].remove(numset[r][c]);
                            }
                        }
                    }
    	        }
    	    }
    	    
    	    // 
    	    for (int r = 0; r < 9; r++) {
    	        for (int c = 0; c < 9; c++) {
    	            if (numset[r][c].isSingletonSet())
    	                continue;
    	            // for vertical line
    	            tmp.fullize();
    	            for (int dr = 0; dr < 9; dr++) {
    	                if (dr == r) continue;
    	                tmp.remove(numset[dr][c]);
    	            }
    	            if (tmp.isSingletonSet()) {
    	                numset[r][c].emptize();
    	                numset[r][c].add(tmp.smallestValue());
    	                continue;
    	            }
    	            
    	            // for hrizontal line
    	            tmp.fullize();
    	            for (int dc = 0; dc < 9; dc++) {
    	                if (dc == c) continue;
    	                tmp.remove(numset[r][dc]);
    	            }
    	            if (tmp.isSingletonSet()) {
    	                numset[r][c].emptize();
    	                numset[r][c].add(tmp.smallestValue());
    	                continue;
    	            }
    	            
    	            // for the same block
    	            tmp.fullize();
    	            for (int dr = 0; dr < 9; dr++) {
    	                for (int dc = 0; dc < 9; dc++) {
    	                    if (dr == r && dc == c) continue;
    	                    if (dr/3 == r/3 && dc/3 == c/3)
    	                        tmp.remove(numset[dr][dc]);
    	                }
    	            }
    	            if (tmp.isSingletonSet()) {
    	                numset[r][c].emptize();
    	                numset[r][c].add(tmp.smallestValue());
    	                continue;
    	            }
    	        }
    	    }

    	    
    	    // set values;
    	    for (int r = 0; r < 9; r++) {
    	        for (int c = 0; c < 9; c++) {
    	            if (! numset[r][c].isSingletonSet()) 
    	                continue;
                    int val;
                    try {
                        val = (new Integer(table[r][c].getText())).intValue();
                    } catch (NumberFormatException ex) {
                        val = 0;
                    }
                    if (val == 0) {
                        table[r][c].setText(String.valueOf(numset[r][c].smallestValue()));
                        updated = true;
                    }
                }
    	    }
    	}
    	
    }
    

	class SymFocus extends java.awt.event.FocusAdapter
	{
		public void focusLost(java.awt.event.FocusEvent event)
		{
			Object object = event.getSource();
			if (object instanceof TextField)
				((TextField)object).select(0,0);
		}

		public void focusGained(java.awt.event.FocusEvent event)
		{
			Object object = event.getSource();
			if (object instanceof TextField)
				((TextField) object).selectAll();
		}
	}

}