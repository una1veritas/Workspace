/*
    ���̸׽�� java.awt.applet.Applet �׽���g���������̂ł�.
 */

import java.awt.*;
import java.applet.*;

import java.net.*;
import java.io.*;

public class TcpClientApplet extends Applet
{
	public void init()
	{
		// symantec.itools.net.RelativeURL �� symantec.itools.awt.util.StatusScroller
		// ���g�p���Ȃ��ꍇ�́A�ȉ��̍s���폜���Ă�������
		//symantec.itools.lang.Context.setApplet(this);
	
		// �ȉ��́C���߰��Ă̐����Ə��������s�����ނ́CVisucalCafe��
		// ����Ď�����������܂����D���̺��ނ�ύX����ꍇ�́C
		// VisualCafe���������麰�ނɉ��������ިݸތ`���ŋL�q���Ă��������D
		// �����Ȃ��ƁCVisualCafe��̫���޻޲ő��ɕύX�𔽉f�����邱��
		// ���ł��Ȃ���������܂���D
		//{{INIT_CONTROLS
		setLayout(null);
		setSize(426,266);
		button1.setActionCommand("button");
		button1.setLabel("���M�I");
		add(button1);
		button1.setBackground(java.awt.Color.lightGray);
		button1.setBounds(276,186,105,26);
		choice1.addItem("Pentium MMX 233 MHz");
		choice1.addItem("AMD K6 266 MHz");
		choice1.addItem("AMD K6-2 3D 295 MHz");
		choice1.addItem("Cyrix MII-300");
		try {
			choice1.select(0);
		}
		catch (IllegalArgumentException e) { }
		add(choice1);
		choice1.setBounds(24,18,169,25);
		panel1.setLayout(null);
		add(panel1);
		panel1.setBackground(java.awt.Color.lightGray);
		panel1.setBounds(24,48,143,93);
		radioButton1.setCheckboxGroup(Group1);
		radioButton1.setState(true);
		radioButton1.setLabel("ASUSTek");
		panel1.add(radioButton1);
		radioButton1.setBounds(6,6,127,21);
		radioButton2.setCheckboxGroup(Group1);
		radioButton2.setLabel("MSI");
		panel1.add(radioButton2);
		radioButton2.setBounds(6,36,120,20);
		radioButton3.setCheckboxGroup(Group1);
		radioButton3.setLabel("EPoX");
		panel1.add(radioButton3);
		radioButton3.setBounds(6,60,123,25);
		add(textField1);
		textField1.setBounds(24,180,180,25);
		label1.setText("���̑�");
		add(label1);
		label1.setBounds(24,156,148,21);
		//}}
	
		//{{REGISTER_LISTENERS
		SymAction lSymAction = new SymAction();
		button1.addActionListener(lSymAction);
		//}}
	}
	
	//{{DECLARE_CONTROLS
	java.awt.Button button1 = new java.awt.Button();
	java.awt.Choice choice1 = new java.awt.Choice();
	java.awt.Panel panel1 = new java.awt.Panel();
	java.awt.Checkbox radioButton1 = new java.awt.Checkbox();
	java.awt.CheckboxGroup Group1 = new java.awt.CheckboxGroup();
	java.awt.Checkbox radioButton2 = new java.awt.Checkbox();
	java.awt.Checkbox radioButton3 = new java.awt.Checkbox();
	java.awt.TextField textField1 = new java.awt.TextField();
	java.awt.Label label1 = new java.awt.Label();
	//}}

	class SymAction implements java.awt.event.ActionListener
	{
		public void actionPerformed(java.awt.event.ActionEvent event)
		{
			Object object = event.getSource();
			if (object == button1)
				button1_Action(event);
		}
	}

	void button1_Action(java.awt.event.ActionEvent event)
	{
		// to do: ���ނ������ɋL�q���܂�.
        int c;
        Socket soc;
        String str ="Hello Net World!!\n";
        int slength;
        OutputStream out;

        try {
            soc = new Socket("localhost",5234);
            out = soc.getOutputStream();
            out.write(choice1.getSelectedItem().getBytes());
            out.write(Group1.getSelectedCheckbox().getLabel().getBytes());
            out.write(textField1.getText().getBytes());
            out.close();
        } catch (IOException x) {
            System.err.println("Caught IOException: "+x);
            System.exit(1);
        }
	}
}
