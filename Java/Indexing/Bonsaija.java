/*
    このｸﾗｽは java.awt.applet.Applet ｸﾗｽを拡張したものです.
 */

import java.awt.*;
import java.applet.*;

import java.io.*;
import java.net.*;

import symantec.itools.awt.util.spinner.NumericSpinner;
import symantec.itools.awt.HorizontalSlider;
//import symantec.itools.awt.TreeView;
import symantec.itools.awt.BorderPanel;

public class Bonsaija extends Applet implements Runnable
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
		setSize(508,300);
		outputArea = new java.awt.TextArea("",0,0,TextArea.SCROLLBARS_VERTICAL_ONLY);
		outputArea.setBounds(18,144,330,138);
		outputArea.setFont(new Font("Dialog", Font.PLAIN, 11));
		add(outputArea);
		treeview = new symantec.itools.awt.TreeView();
		treeview.setLayout(null);
		treeview.setBounds(360,138,134,144);
		add(treeview);
		treeview.setCursor(new Cursor(Cursor.HAND_CURSOR));
		label5 = new java.awt.Label("Decision Tree:");
		label5.setBounds(366,114,84,24);
		add(label5);
		runButton = new java.awt.Button();
		runButton.setActionCommand("button");
		runButton.setLabel("Run");
		runButton.setBounds(24,114,85,25);
		runButton.setBackground(new Color(12632256));
		add(runButton);
		buttonStop = new java.awt.Button();
		buttonStop.setActionCommand("button");
		buttonStop.setLabel("Stop");
		buttonStop.setBounds(126,114,84,24);
		buttonStop.setBackground(new Color(12632256));
		add(buttonStop);
		buttonStop.setEnabled(false);
		label1 = new java.awt.Label("Example (+)",Label.RIGHT);
		label1.setBounds(84,12,66,18);
		add(label1);
		label2 = new java.awt.Label("Counter-example (-)",Label.RIGHT);
		label2.setBounds(36,36,114,22);
		add(label2);
		exampleNameField = new java.awt.TextField();
		exampleNameField.setBounds(156,12,336,20);
		add(exampleNameField);
		counterNameField = new java.awt.TextField();
		counterNameField.setBounds(156,36,336,20);
		add(counterNameField);
		trainningSpinner = new symantec.itools.awt.util.spinner.NumericSpinner();
		try {
			trainningSpinner.setMin(3);
		}
		catch(java.beans.PropertyVetoException e) { }
		try {
			trainningSpinner.setCurrent(10);
		}
		catch(java.beans.PropertyVetoException e) { }
		trainningSpinner.setLayout(null);
		try {
			trainningSpinner.setMax(40);
		}
		catch(java.beans.PropertyVetoException e) { }
		trainningSpinner.setBounds(312,66,54,24);
		add(trainningSpinner);
		ratioSpinner = new symantec.itools.awt.util.spinner.NumericSpinner();
		try {
			ratioSpinner.setMin(1);
		}
		catch(java.beans.PropertyVetoException e) { }
		try {
			ratioSpinner.setCurrent(30);
		}
		catch(java.beans.PropertyVetoException e) { }
		ratioSpinner.setLayout(null);
		try {
			ratioSpinner.setMax(100);
		}
		catch(java.beans.PropertyVetoException e) { }
		ratioSpinner.setBounds(156,66,60,24);
		add(ratioSpinner);
		label3 = new java.awt.Label("Training Size:",Label.RIGHT);
		label3.setBounds(228,66,80,21);
		add(label3);
		label4 = new java.awt.Label("Overall sampling ratio:",Label.RIGHT);
		label4.setBounds(24,66,126,22);
		add(label4);
		getExampleButton = new java.awt.Button();
		getExampleButton.setActionCommand("button");
		getExampleButton.setLabel("Get / Resample");
		getExampleButton.setBounds(378,66,108,24);
		getExampleButton.setBackground(new Color(12632256));
		add(getExampleButton);
		label6 = new java.awt.Label("URL",Label.RIGHT);
		label6.setBounds(12,18,30,24);
		add(label6);
		//}}
	
		//{{REGISTER_LISTENERS
		SymAction lSymAction = new SymAction();
		runButton.addActionListener(lSymAction);
		getExampleButton.addActionListener(lSymAction);
		buttonStop.addActionListener(lSymAction);
		treeview.addActionListener(lSymAction);
		//}}
	}
	
	//{{DECLARE_CONTROLS
	java.awt.TextArea outputArea;
	symantec.itools.awt.TreeView treeview;
	java.awt.Label label5;
	java.awt.Button runButton;
	java.awt.Button buttonStop;
	java.awt.Label label1;
	java.awt.Label label2;
	java.awt.TextField exampleNameField;
	java.awt.TextField counterNameField;
	symantec.itools.awt.util.spinner.NumericSpinner trainningSpinner;
	symantec.itools.awt.util.spinner.NumericSpinner ratioSpinner;
	java.awt.Label label3;
	java.awt.Label label4;
	java.awt.Button getExampleButton;
	java.awt.Label label6;
	//}}
        
	class SymAction implements java.awt.event.ActionListener
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
	
	
	// declare contributing variables
	Bonsai bonsai;
    Thread thread;
	
	public void start() {
	    /*if (thread == null) {
	        thread = new Thread(this);
	        thread.start();
	    }*/
	}
	
	public void run() {
        AlphabetIndexing neighbor;
        DecisionTree tree;
        double currEval, neighborEval;
        
        while (true) {
            neighbor = bonsai.findBetterNeighbor();
            if (neighbor == null) {
                break;
            }
            bonsai.setIndexing(neighbor);
            bonsai.setDecisionTree(bonsai.makeDecisionTree(bonsai.indexing()));
            outputArea.append(bonsai.indexing().translate(neighbor.domainString()) 
					+ " " + Double.toString(bonsai.verifyDecisionTree(bonsai.decisionTree(), bonsai.indexing())).substring(0,4));
            outputArea.append("\n   " + bonsai.decisionTree().toString() + "\n");
            treeview.setTreeStructure(bonsai.decisionTree().structureString());
            treeview.triggerRedraw();
            //repaint();
            /*try {
                Thread.sleep(100);
            } catch (InterruptedException e) { }
            */
        }
        treeview.setTreeStructure(bonsai.decisionTree().structureString());
        treeview.triggerRedraw();
        outputArea.append("Result: \n"+bonsai.indexing().domainString()+"\n"+bonsai.indexing().translate(bonsai.indexing().domainString()) + "\n");
        outputArea.append(bonsai.decisionTree().toString()+" with accuracy "+String.valueOf(bonsai.verifyDecisionTree()).substring(0,4) + "\n");
        outputArea.append("finished all computation.\n");
    }

	public void stop() {
	    if (thread != null) {
	        thread.stop();
	        thread = null;
	    }
	    getExampleButton.setEnabled(true);
	    runButton.setLabel("Run");
	    runButton.setEnabled(true);
	    buttonStop.setEnabled(false);
	}
	
	void runButton_Action(java.awt.event.ActionEvent event)
	{
		// to do: ｺｰﾄﾞをここに記述します.
        
        if (bonsai == null || (! bonsai.hasExamples())) {
            outputArea.setText("No examples have been supplied yet;\n");
            outputArea.append("Do \"Get/Resample\" first.\n");
            return;
        }
        
		getExampleButton.setEnabled(false);
		runButton.setLabel("Running...");
		runButton.setEnabled(false);
		buttonStop.setEnabled(true);
		
	    if (thread == null) {
	        thread = new Thread(this);
	        thread.start();
	    }
	}

	void getExampleButton_Action(java.awt.event.ActionEvent event)
	{
		// to do: ｺｰﾄﾞをここに記述します.
        BufferedReader pr, nr;
        
        try {
            pr = new BufferedReader(new InputStreamReader((new URL(exampleNameField.getText())).openStream()));
            nr = new BufferedReader(new InputStreamReader((new URL(counterNameField.getText())).openStream()));
        } catch(IOException exc) {
            outputArea.append("Error in opening stream: " + exc + ".\nCheck whether URL address is correct.\n");
            //System.out.println("Error :" + exc + "\n");
            return;
        }

		getExampleButton.setEnabled(false);
        bonsai = new Bonsai(pr, nr, ((double)Integer.valueOf(ratioSpinner.getCurrentText()).intValue())/100, Integer.valueOf(trainningSpinner.getCurrentText()).intValue());
        outputArea.setText(bonsai.toString());
        outputArea.append("\n");
		getExampleButton.setEnabled(true);
	}

	void buttonStop_Action(java.awt.event.ActionEvent event)
	{
		// to do: ｺｰﾄﾞをここに記述します.		
		stop();
	}
	
}
