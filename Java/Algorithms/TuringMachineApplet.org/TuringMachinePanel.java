// Decompiled by Jad v1.5.7d. Copyright 2000 Pavel Kouznetsov.
// Jad home page: http://www.geocities.com/SiliconValley/Bridge/8617/jad.html
// Decompiler options: packimports(3) 
// Source File Name:   xTuringMachinePanel.java

package tm;

import java.awt.*;
import java.net.URL;

// Referenced classes of package tmcm.xTuringMachine:
//            Machine, MachinePanel

public class TuringMachinePanel extends Panel
{

    public TuringMachinePanel(URL aurl[], String as[])
    {
        setBackground(Color.lightGray);
        setLayout(new BorderLayout(5, 5));
        machineChoice = new Choice();
        machineChoice.addItem("[New]");
        if(aurl == null || aurl.length == 0)
        {
            machineChoice.addItem("Untitled 1");
            machine = new MachinePanel(this);
            untitledCt = 1;
        } else
        {
            for(int i = 0; i < as.length; i++)
                machineChoice.addItem(as[i]);

            machine = new MachinePanel(aurl, this);
            untitledCt = 0;
        }
        currentChoice = 1;
        machineChoice.select(1);
        add("North", machineChoice);
        add("Center", machine);
    }

    public TuringMachinePanel(String as[])
    {
        setBackground(Color.lightGray);
        setLayout(new BorderLayout(5, 5));
        machineChoice = new Choice();
        machineChoice.addItem("[New]");
        if(as == null || as.length == 0)
        {
            machineChoice.addItem("Untitled 1");
            machine = new MachinePanel(this);
            untitledCt = 1;
        } else
        {
            for(int i = 0; i < as.length; i++)
                machineChoice.addItem(as[i]);

            machine = new MachinePanel(as, this);
            untitledCt = 0;
        }
        currentChoice = 1;
        machineChoice.select(1);
        add("North", machineChoice);
        add("Center", machine);
    }

    public Insets insets()
    {
        return new Insets(5, 5, 5, 5);
    }

    public void start()
    {
    }

    public void stop()
    {
        machine.TM.stopRunning();
    }

    public void destroy()
    {
        if(machine.TM.runner != null && machine.TM.runner.isAlive())
            machine.TM.runner.stop();
        if(machine.timer != null && machine.timer.isAlive())
            machine.timer.stop();
        if(machine.loaders != null)
        {
            for(int i = 0; i < machine.loaders.length; i++)
                if(machine.loaders[i] != null && machine.loaders[i].isAlive())
                    machine.loaders[i].stop();

        }
    }

    void fileLoaded(String s)
    {
        machineChoice.addItem(s);
        currentChoice = machineChoice.countItems() - 1;
        machineChoice.select(currentChoice);
    }

    void doMachineChoice()
    {
        int i = machineChoice.getSelectedIndex();
        if(i == currentChoice)
            return;
        if(i > 0)
        {
            machine.selectMachineNumber(i - 1);
            currentChoice = i;
        } else
        {
            int j = machineChoice.countItems();
            untitledCt++;
            machineChoice.addItem("Untitled " + untitledCt);
            machineChoice.select(j);
            currentChoice = j;
            machine.doNewMachineCommand(j - 1);
        }
    }

    public boolean action(Event event, Object obj)
    {
        if(event.target == machineChoice)
        {
            doMachineChoice();
            return true;
        } else
        {
            return super.action(event, obj);
        }
    }

    MachinePanel machine;
    Choice machineChoice;
    int currentChoice;
    int untitledCt;
}
