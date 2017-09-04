/*
    このｸﾗｽは java.awt.applet.Applet ｸﾗｽを拡張したものです.
 */

import java.awt.*;
import java.applet.*;
import java.io.*;
import java.net.*;

import symantec.itools.awt.util.spinner.NumericSpinner;
import symantec.itools.awt.HorizontalSlider;

import DecisionTree;
import SuffixTreeEnumerator;
import AlphabetIndexingNeighbor;
import symantec.itools.awt.TreeView;

public class Bonsaija extends Applet 
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
		setSize(456,304);
		outputArea.setEditable(false);
		add(outputArea);
		outputArea.setFont(new Font("Dialog", Font.PLAIN, 11));
		outputArea.setBounds(12,198,300,90);
		runButton.setActionCommand("button");
		runButton.setLabel("Run");
		add(runButton);
		runButton.setBackground(java.awt.Color.lightGray);
		runButton.setBounds(18,150,85,25);
		buttonStop.setActionCommand("button");
		buttonStop.setLabel("Stop");
		buttonStop.setEnabled(false);
		add(buttonStop);
		buttonStop.setBackground(java.awt.Color.lightGray);
		buttonStop.setBounds(120,150,84,24);
		label1.setText("Class +");
		label1.setAlignment(java.awt.Label.RIGHT);
		add(label1);
		label1.setBounds(18,30,51,18);
		label2.setText("Class -");
		label2.setAlignment(java.awt.Label.RIGHT);
		add(label2);
		label2.setBounds(18,54,51,18);
		exampleNameField.setText("http://w3.donald.ai.kyutech.ac.jp/~sin/demos/bonsai/memb.pos");
		add(exampleNameField);
		exampleNameField.setBounds(78,30,366,18);
		counterNameField.setText("http://w3.donald.ai.kyutech.ac.jp/~sin/demos/bonsai/memb.neg");
		add(counterNameField);
		counterNameField.setBounds(78,54,366,18);
		label3.setText("Training examples");
		label3.setAlignment(java.awt.Label.CENTER);
		add(label3);
		label3.setBounds(132,84,108,18);
		label4.setText("Sampling ratio (%)");
		label4.setAlignment(java.awt.Label.CENTER);
		add(label4);
		label4.setBounds(12,84,108,18);
		getExampleButton.setActionCommand("button");
		getExampleButton.setLabel("Get/Resample");
		add(getExampleButton);
		getExampleButton.setBackground(java.awt.Color.lightGray);
		getExampleButton.setBounds(336,108,102,24);
		label6.setText("Example source URL:");
		add(label6);
		label6.setBounds(12,6,128,15);
		domainTextArea.setEditable(false);
		domainTextArea.setEnabled(false);
		add(domainTextArea);
		domainTextArea.setBounds(12,180,300,18);
		label7.setText("Index labels");
		label7.setAlignment(java.awt.Label.CENTER);
		add(label7);
		label7.setBounds(252,84,72,18);
		indexChoice.addItem("ab");
		indexChoice.addItem("abc");
		indexChoice.addItem("abcd");
		try {
			indexChoice.select(1);
		}
		catch (IllegalArgumentException e) { }
		add(indexChoice);
		indexChoice.setBounds(252,108,76,25);
		try {
			trainningSpinner.setCurrent(10);
		}
		catch(java.beans.PropertyVetoException e) { }
		try {
			trainningSpinner.setMax(20);
		}
		catch(java.beans.PropertyVetoException e) { }
		add(trainningSpinner);
		trainningSpinner.setBounds(144,108,84,24);
		try {
			ratioSpinner.setMin(1);
		}
		catch(java.beans.PropertyVetoException e) { }
		try {
			ratioSpinner.setCurrent(30);
		}
		catch(java.beans.PropertyVetoException e) { }
		try {
			ratioSpinner.setMax(100);
		}
		catch(java.beans.PropertyVetoException e) { }
		add(ratioSpinner);
		ratioSpinner.setBounds(24,108,84,24);
		add(treeview);
		treeview.setBounds(324,168,120,120);
		//}}
	
		//{{REGISTER_LISTENERS
		SymAction lSymAction = new SymAction();
		runButton.addActionListener(lSymAction);
		getExampleButton.addActionListener(lSymAction);
		buttonStop.addActionListener(lSymAction);
		//}}
	}
	
	//{{DECLARE_CONTROLS
	java.awt.TextArea outputArea = new java.awt.TextArea();
	java.awt.Button runButton = new java.awt.Button();
	java.awt.Button buttonStop = new java.awt.Button();
	java.awt.Label label1 = new java.awt.Label();
	java.awt.Label label2 = new java.awt.Label();
	java.awt.TextField exampleNameField = new java.awt.TextField();
	java.awt.TextField counterNameField = new java.awt.TextField();
	java.awt.Label label3 = new java.awt.Label();
	java.awt.Label label4 = new java.awt.Label();
	java.awt.Button getExampleButton = new java.awt.Button();
	java.awt.Label label6 = new java.awt.Label();
	java.awt.TextArea domainTextArea = new java.awt.TextArea("",0,0,TextArea.SCROLLBARS_NONE);
	java.awt.Label label7 = new java.awt.Label();
	java.awt.Choice indexChoice = new java.awt.Choice();
	symantec.itools.awt.util.spinner.NumericSpinner trainningSpinner = new symantec.itools.awt.util.spinner.NumericSpinner();
	symantec.itools.awt.util.spinner.NumericSpinner ratioSpinner = new symantec.itools.awt.util.spinner.NumericSpinner();
	symantec.itools.awt.TreeView treeview = new symantec.itools.awt.TreeView();
	//}}
        
	public class SymAction implements java.awt.event.ActionListener
	{
		public void actionPerformed(java.awt.event.ActionEvent event)
		{
			Object object = event.getSource();
			if (object == runButton)
				runButton_Action(event);
			else if (object == getExampleButton)
				getExampleButton_Action(event);
			else if (object == buttonStop)
				buttonStop_Action(event);
		}
	}
	
	public class SearchThread extends Thread {
	    private boolean searching;
        
    	public void run() {
    		getExampleButton.setEnabled(false);
    		runButton.setLabel("Running...");
    		runButton.setEnabled(false);
    		buttonStop.setEnabled(true);
            
            search();
            
            getExampleButton.setEnabled(true);
    	    runButton.setLabel("Run");
    	    runButton.setEnabled(true);
    	    buttonStop.setEnabled(false);
    	}
    	
        public void search() {
            AlphabetIndexing neighbor;
            DecisionTree tree;
            double currEval, neighborEval;
            
            bonsai.indexing().randomize();
            treeview.setTreeStructure(bonsai.decisionTree().structureString());
            treeview.triggerRedraw();
            outputArea.setText(bonsai.indexing().translate(bonsai.indexing().domainString()) 
					+ " " + Double.toString(bonsai.verifyDecisionTree()).substring(0,4) + "  "+ bonsai.decisionTree());            //
            while (true) {
                neighbor = bonsai.findBetterNeighbor();
                if (neighbor == null) {
                    break;
                }
                bonsai.setIndexing(neighbor);
                bonsai.renewDecisionTree();
                //
                outputArea.insert(bonsai.indexing().translate(neighbor.domainString()) 
    					+ " " + Double.toString(bonsai.verifyDecisionTree(bonsai.decisionTree(), bonsai.indexing())).substring(0,4)+"  " + bonsai.decisionTree() + "\n", 0);
                treeview.setEnabled(false);
                treeview.setTreeStructure(bonsai.decisionTree().structureString());
                treeview.triggerRedraw();
                treeview.setEnabled(true);
            }
            treeview.setTreeStructure(bonsai.decisionTree().structureString());
            treeview.triggerRedraw();
            outputArea.insert(bonsai.indexing().translate(bonsai.indexing().domainString()) +" with accuracy "+String.valueOf(bonsai.verifyDecisionTree()).substring(0,4) + "\n", 0);
        }
    }
	
	
	// declare contributing variables
	Bonsai bonsai;
	SearchThread searcher;
	
	void runButton_Action(java.awt.event.ActionEvent event)
	{
		// to do: ｺｰﾄﾞをここに記述します.
        
        if (bonsai == null || (! bonsai.hasExamples())) {
            outputArea.setText("No examples have been supplied yet;\n");
            outputArea.append("Do \"Get/Resample\" first.\n");
            return;
        }
        if (searcher != null) searcher.stop();
		searcher = new SearchThread();
		searcher.start();
	}

	void getExampleButton_Action(java.awt.event.ActionEvent event)
	{
		// to do: ｺｰﾄﾞをここに記述します.
        BufferedReader pr, nr;
        
        try {
    		getExampleButton.setEnabled(false);
            pr = new BufferedReader(new InputStreamReader((new URL(exampleNameField.getText())).openStream()));
            nr = new BufferedReader(new InputStreamReader((new URL(counterNameField.getText())).openStream()));
    		getExampleButton.setEnabled(true);
        } catch(IOException exc) {
            outputArea.append("Error on reading stream: " + exc + ".\nCheck whether URL address is correct.\n");
            return;
        }
        bonsai = new Bonsai(indexChoice.getSelectedItem(), pr, nr, ((double)Integer.valueOf(ratioSpinner.getCurrentText()).intValue())/100, Integer.valueOf(trainningSpinner.getCurrentText()).intValue());
        domainTextArea.setText(bonsai.indexing().domainString());
        outputArea.setText(bonsai.toString());
        outputArea.append("\n");
	}

	void buttonStop_Action(java.awt.event.ActionEvent event)
	{
		// to do: ｺｰﾄﾞをここに記述します.
		stop_bonsai();
	}
	
	void stop_bonsai() {
	    if (searcher != null) 
	        searcher.stop();
	    getExampleButton.setEnabled(true);
	    runButton.setLabel("Run");
	    runButton.setEnabled(true);
	    buttonStop.setEnabled(false);
	}
	
}
