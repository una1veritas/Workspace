// Decompiled by Jad v1.5.7d. Copyright 2000 Pavel Kouznetsov.
// Jad home page: http://www.geocities.com/SiliconValley/Bridge/8617/jad.html
// Decompiler options: packimports(3) 
// Source File Name:   MachinePanel.java

package tm;

import java.awt.*;
import java.io.*;
import java.net.URL;
import java.util.Vector;

// Referenced classes of package tmcm.xTuringMachine:
//            Machine, MachineData, MachineInputException, MachineLoader, 
//            Palette, RuleCanvas, RuleMakerCanvas, RuleTable, 
//            xTuringMachinePanel

class MachinePanel extends Panel
    implements Runnable
{

    MachinePanel(TuringMachinePanel xturingmachinepanel)
    {
        canLoad = true;
        canSave = true;
        currentData = null;
        currentMachineNumber = 0;
        timeToAlarm = 0;
        currentFocus = 0;
        currentPaletteDisplay = 0;
        owner = xturingmachinepanel;
        setUp();
        currentData = new MachineData();
        machineData.addElement(currentData);
        TM.setMachineData(currentData, 0);
        ruleTable.setMachineData(currentData);
        ruleMaker.setMachineData(currentData);
    }

    MachinePanel(URL aurl[], TuringMachinePanel xturingmachinepanel)
    {
        canLoad = true;
        canSave = true;
        currentData = null;
        currentMachineNumber = 0;
        timeToAlarm = 0;
        currentFocus = 0;
        currentPaletteDisplay = 0;
        owner = xturingmachinepanel;
        setUp();
        loaders = new MachineLoader[aurl.length];
        for(int i = 0; i < aurl.length; i++)
        {
            machineData.addElement(null);
            loaders[i] = new MachineLoader(aurl[i], this, i);
        }

        setControlsForLoading();
        TM.setMachineData(null, 0);
        TM.setMessage("Loading from URL:", aurl[0].toString(), null);
        ruleTable.setMachineData(null);
        ruleMaker.setMachineData(null);
    }

    MachinePanel(String as[], TuringMachinePanel xturingmachinepanel)
    {
        canLoad = true;
        canSave = true;
        currentData = null;
        currentMachineNumber = 0;
        timeToAlarm = 0;
        currentFocus = 0;
        currentPaletteDisplay = 0;
        owner = xturingmachinepanel;
        setUp();
        loaders = new MachineLoader[as.length];
        for(int i = 0; i < as.length; i++)
        {
            machineData.addElement(null);
            loaders[i] = new MachineLoader(null, as[i], this, i);
        }

        setControlsForLoading();
        TM.setMachineData(null, 0);
        TM.setMessage("Loading from file:", as[0], null);
        ruleTable.setMachineData(null);
        ruleMaker.setMachineData(null);
    }

    private void setUp()
    {
        setLayout(new BorderLayout(5, 5));
        setBackground(Color.lightGray);
        machineData = new Vector();
        runButton = new Button("Run");
        stepButton = new Button("Step");
        clearTapeButton = new Button("Clear Tape");
        deleteRuleButton = new Button("Delete Rule");
        deleteRuleButton.disable();
        loadButton = new Button("Load File");
        saveButton = new Button("Save");
        makeButton = new Button("Make Rule");
        speedChoice = new Choice();
        speedChoice.addItem("Fastest");
        speedChoice.addItem("Fast");
        speedChoice.addItem("Moderate");
        speedChoice.addItem("Slow");
        speedChoice.addItem("Slowest");
        speedChoice.select("Moderate");
        currentSpeed = 2;
        TM = new Machine(this);
        TM.speed = currentSpeed;
        machineData = new Vector();
        palette = new Palette(this);
        ruleTable = new RuleTable(this);
        ruleMaker = new RuleMakerCanvas(this);
        Panel panel = new Panel();
        panel.setLayout(new GridLayout(7, 1, 5, 5));
        panel.add(speedChoice);
        panel.add(runButton);
        panel.add(stepButton);
        panel.add(clearTapeButton);
        panel.add(deleteRuleButton);
        panel.add(loadButton);
        panel.add(saveButton);
        Panel panel1 = new Panel();
        panel1.setLayout(new BorderLayout(5, 5));
        panel1.add("Center", ruleMaker);
        panel1.add("East", makeButton);
        Panel panel2 = new Panel();
        panel2.setLayout(new GridLayout(2, 1, 5, 5));
        panel2.add(palette);
        panel2.add(panel1);
        Panel panel3 = new Panel();
        panel3.setLayout(new BorderLayout(5, 5));
        panel3.add("North", panel2);
        panel3.add("Center", ruleTable);
        add("North", TM);
        add("Center", panel3);
        add("West", panel);
    }

    synchronized void doNewMachineCommand(int i)
    {
        TM.stopRunning();
        dropFocus(currentFocus);
        if(currentData != null)
            currentData.saveCurrentSquare = TM.currentSquare;
        else
            setNormalControls();
        currentData = new MachineData();
        machineData.addElement(currentData);
        currentMachineNumber = machineData.size() - 1;
        TM.setMachineData(currentData, 0);
        ruleMaker.setMachineData(currentData);
        ruleTable.setMachineData(currentData);
    }

    synchronized void selectMachineNumber(int i)
    {
        TM.clearTemporaryMessage();
        if(i < 0 || i >= machineData.size() || i == currentMachineNumber)
            return;
        dropFocus(currentFocus);
        TM.stopRunning();
        if(currentData != null)
            currentData.saveCurrentSquare = TM.currentSquare;
        currentData = (MachineData)machineData.elementAt(i);
        if(currentData == null)
            setControlsForLoading();
        else
            setNormalControls();
        currentMachineNumber = i;
        if(currentData == null)
        {
            TM.setMachineData(null, 0);
            ruleMaker.setMachineData(null);
            ruleTable.setMachineData(null);
            if(loaders[i].url != null)
                TM.setMessage("Error while oading from URL:", loaders[i].url.toString(), loaders[i].errorMessage);
            else
                TM.setMessage("Error while loading from file:", loaders[i].fileName, loaders[i].errorMessage);
        } else
        {
            TM.setMachineData(currentData, currentData.saveCurrentSquare);
            ruleMaker.setMachineData(currentData);
            ruleTable.setMachineData(currentData);
        }
    }

    void requestFocus(int i, int j)
    {
        TM.clearTemporaryMessage();
        if(i != currentFocus)
            switch(currentFocus)
            {
            case 1: // '\001'
                ruleTable.canvas.focusIsOff();
                break;

            case 3: // '\003'
                TM.focusIsOff();
                break;

            case 2: // '\002'
                ruleMaker.focusIsOff();
                break;
            }
        currentFocus = i;
        currentPaletteDisplay = j;
        if(focused && currentFocus != 0)
        {
            palette.setDisplay(j);
            switch(currentFocus)
            {
            case 1: // '\001'
                ruleTable.canvas.focusIsOn();
                break;

            case 3: // '\003'
                TM.focusIsOn();
                break;

            case 2: // '\002'
                ruleMaker.focusIsOn();
                break;
            }
        }
        if(!focused)
            requestFocus();
    }

    void dropFocus(int i)
    {
        if(i == currentFocus)
        {
            currentPaletteDisplay = 0;
            palette.setDisplay(0);
            switch(currentFocus)
            {
            case 1: // '\001'
                ruleTable.canvas.focusIsOff();
                break;

            case 3: // '\003'
                TM.focusIsOff();
                break;

            case 2: // '\002'
                ruleMaker.focusIsOff();
                break;
            }
            currentFocus = 0;
        }
    }

    public synchronized boolean gotFocus(Event event, Object obj)
    {
        if(timeToAlarm > 0)
        {
            setAlarm(0);
        } else
        {
            focused = true;
            switch(currentFocus)
            {
            case 1: // '\001'
                ruleTable.canvas.focusIsOn();
                break;

            case 3: // '\003'
                TM.focusIsOn();
                break;

            case 2: // '\002'
                ruleMaker.focusIsOn();
                break;
            }
            if(currentFocus != 0)
                palette.setDisplay(currentPaletteDisplay);
        }
        return true;
    }

    public synchronized boolean lostFocus(Event event, Object obj)
    {
        setAlarm(200);
        return true;
    }

    synchronized void alarm()
    {
        focused = false;
        switch(currentFocus)
        {
        case 1: // '\001'
            ruleTable.canvas.focusIsOff();
            break;

        case 3: // '\003'
            TM.focusIsOff();
            break;

        case 2: // '\002'
            ruleMaker.focusIsOff();
            break;
        }
        if(currentFocus != 0)
            palette.setDisplay(0);
    }

    public boolean keyDown(Event event, int i)
    {
        TM.clearTemporaryMessage();
        if(i == 9)
            switch(currentFocus)
            {
            case 0: // '\0'
            case 3: // '\003'
                ruleMaker.selectedColumn = 0;
                requestFocus(2, 1);
                break;

            case 1: // '\001'
                if(running)
                {
                    ruleMaker.selectedColumn = 0;
                    requestFocus(2, 1);
                } else
                {
                    TM.selectItem(TM.currentSquare);
                }
                break;

            case 2: // '\002'
                if(currentData != null && currentData.getRuleCount() > 0)
                {
                    if(ruleTable.canvas.selectedRule < 0)
                        ruleTable.canvas.select(0, 2);
                    else
                        ruleTable.canvas.select(ruleTable.canvas.selectedRule, 2);
                    requestFocus(1, 5);
                } else
                if(!running)
                    TM.selectItem(TM.currentSquare);
                break;
            }
        else
            switch(currentFocus)
            {
            case 1: // '\001'
                ruleTable.canvas.processKey(i);
                break;

            case 3: // '\003'
                TM.processKey(i);
                break;

            case 2: // '\002'
                ruleMaker.processKey(i);
                break;
            }
        return true;
    }

    void doPaletteClick(int i)
    {
        switch(currentFocus)
        {
        case 1: // '\001'
            ruleTable.canvas.processPaletteClick(i, palette.display);
            break;

        case 3: // '\003'
            TM.processPaletteClick(i, palette.display);
            break;

        case 2: // '\002'
            ruleMaker.processPaletteClick(i, palette.display);
            break;
        }
    }

    synchronized void setAlarm(int i)
    {
        timeToAlarm = i;
        if(timer == null || !timer.isAlive() && i > 0)
        {
            timer = new Thread(this);
            timer.start();
        } else
        {
            notify();
        }
    }

    public void run()
    {
        do
        {
            synchronized(this)
            {
                while(timeToAlarm <= 0) 
                    try
                    {
                        wait();
                    }
                    catch(InterruptedException _ex) { }
            }
            synchronized(this)
            {
                try
                {
                    wait(timeToAlarm);
                }
                catch(InterruptedException _ex) { }
            }
            synchronized(this)
            {
                if(timeToAlarm > 0)
                {
                    timeToAlarm = 0;
                    alarm();
                }
            }
        } while(true);
    }

    void setControlsForLoading()
    {
        runButton.disable();
        stepButton.disable();
        clearTapeButton.setLabel("Clear");
        clearTapeButton.enable();
        deleteRuleButton.disable();
        saveButton.disable();
    }

    void setNormalControls()
    {
        runButton.enable();
        stepButton.enable();
        clearTapeButton.setLabel("Clear Tape");
        clearTapeButton.enable();
        if(canSave)
            saveButton.enable();
    }

    synchronized void loadingError(int i)
    {
        if(loaders[i] != null && i == currentMachineNumber)
            if(loaders[i].url != null)
                TM.setMessage("Error while loading from URL:", loaders[i].url.toString(), loaders[i].errorMessage);
            else
                TM.setMessage("Error while loading from file:", loaders[i].fileName, loaders[i].errorMessage);
    }

    synchronized void doneLoading(MachineData machinedata, int i)
    {
        if(machineData.elementAt(i) != null)
            return;
        machineData.setElementAt(machinedata, i);
        if(i == currentMachineNumber)
        {
            currentData = machinedata;
            TM.setMachineData(currentData, currentData.saveCurrentSquare);
            ruleTable.setMachineData(currentData);
            ruleMaker.setMachineData(currentData);
            setNormalControls();
        }
    }

    synchronized void abortLoad()
    {
        currentData = new MachineData();
        loaders[currentMachineNumber] = null;
        machineData.setElementAt(currentData, currentMachineNumber);
        TM.setMachineData(currentData, 0);
        ruleMaker.setMachineData(currentData);
        ruleTable.setMachineData(currentData);
        setNormalControls();
    }

    void doLoad()
    {
        if(!canLoad)
            return;
        TM.stopRunning();
        FileDialog filedialog = null;
        try
        {
            Object obj = this;
            do
            {
                Container container = ((Component) (obj)).getParent();
                if(container == null)
                    break;
                obj = container;
            } while(true);
            if(!(obj instanceof Frame))
                obj = null;
            filedialog = new FileDialog((Frame)obj, "Select File to Load", 0);
            filedialog.show();
        }
        catch(AWTError _ex)
        {
            TM.setTemporaryMessage("ERROR while trying to create a file dialog box.", "It will not be possible to load files.", null);
            canLoad = false;
            loadButton.disable();
            return;
        }
        catch(RuntimeException _ex)
        {
            TM.setTemporaryMessage("ERROR while trying to create a file dialog box.", "It will not be possible to load files.", null);
            canLoad = false;
            loadButton.disable();
            return;
        }
        String s = filedialog.getFile();
        if(s == null)
            return;
        String s1 = filedialog.getDirectory();
        FileInputStream fileinputstream = null;
        MachineData machinedata;
        try
        {
            fileinputstream = new FileInputStream(new File(s1, s));
            machinedata = new MachineData();
            machinedata.read(fileinputstream);
        }
        catch(MachineInputException machineinputexception)
        {
            TM.setTemporaryMessage("LOAD FAILED:", machineinputexception.getMessage(), null);
            machinedata = null;
        }
        catch(SecurityException securityexception)
        {
            TM.setTemporaryMessage("LOAD FAILED, SECURITY ERROR:", securityexception.getMessage(), null);
            machinedata = null;
        }
        catch(Exception exception1)
        {
            TM.setTemporaryMessage("LOAD FAILED, ERROR:", exception1.toString(), null);
            machinedata = null;
        }
        finally
        {
            if(fileinputstream != null)
                try
                {
                    fileinputstream.close();
                }
                catch(IOException _ex) { }
        }
        if(machinedata != null)
            synchronized(this)
            {
                TM.stopRunning();
                dropFocus(currentFocus);
                owner.fileLoaded(s);
                machineData.addElement(machinedata);
                if(currentData == null)
                    setNormalControls();
                else
                    currentData.saveCurrentSquare = TM.currentSquare;
                currentData = machinedata;
                currentMachineNumber = machineData.size() - 1;
                TM.setMachineData(currentData, currentData.saveCurrentSquare);
                ruleMaker.setMachineData(currentData);
                ruleTable.setMachineData(currentData);
            }
    }

    void doSave()
    {
        if(currentData == null || !canSave)
            return;
        TM.stopRunning();
        String s = null;
        Object obj = null;
        try
        {
            FileDialog filedialog = null;
            try
            {
                Object obj1 = this;
                do
                {
                    Container container = ((Component) (obj1)).getParent();
                    if(container == null)
                        break;
                    obj1 = container;
                } while(true);
                if(!(obj1 instanceof Frame))
                    obj1 = null;
                filedialog = new FileDialog((Frame)obj1, "Save as:", 1);
                filedialog.show();
            }
            catch(AWTError _ex)
            {
                TM.setTemporaryMessage("ERROR while trying to create a file dialog box.", "It will not be possible to save files.", null);
                canSave = false;
                saveButton.disable();
                return;
            }
            catch(RuntimeException _ex)
            {
                TM.setTemporaryMessage("ERROR while trying to create a file dialog box.", "It will not be possible to save files.", null);
                canSave = false;
                saveButton.disable();
                return;
            }
            s = filedialog.getFile();
            if(s == null)
                return;
            String s1 = filedialog.getDirectory();
            PrintStream printstream = new PrintStream(new FileOutputStream(new File(s1, s)));
            currentData.write(printstream, TM.currentSquare);
            printstream.close();
            if(printstream.checkError())
                throw new IOException("Error occurred while writing data.");
        }
        catch(IOException ioexception)
        {
            TM.setTemporaryMessage("OUTPUT ERROR", "while trying to save to the file \"" + s + "\":", ioexception.getMessage());
        }
        catch(SecurityException securityexception)
        {
            TM.setTemporaryMessage("SECURITY ERROR", "while trying to save to the file \"" + s + "\":", securityexception.getMessage());
        }
    }

    void doMakeRule()
    {
        if(currentData == null)
            return;
        if(ruleMaker.state != 999)
        {
            boolean flag = currentData.ruleDefined(ruleMaker.state, ruleMaker.symbol) ^ true;
            currentData.setActionData(ruleMaker.state, ruleMaker.symbol, ruleMaker.newSymbol, ruleMaker.direction, ruleMaker.newState);
            if(flag)
                ruleTable.ruleAdded(ruleMaker.state, ruleMaker.symbol);
            else
                ruleTable.ruleChanged(ruleMaker.state, ruleMaker.symbol);
            if(TM.getStatus() == 1 && currentData.getNewState(TM.machineState, currentData.getTape(TM.machineState)) != 999)
            {
                TM.setChangedAll();
                TM.setStatus(0);
                TM.repaint();
            }
            ruleMaker.ruleMade();
        }
    }

    synchronized void doneRunning(boolean flag)
    {
        runButton.setLabel("Run");
        if(flag)
            stepButton.setLabel("Reset");
        stepButton.enable();
        halted = flag;
        runButton.enable();
        clearTapeButton.enable();
        running = false;
    }

    synchronized void doneStep(boolean flag)
    {
        runButton.enable();
        if(flag)
            stepButton.setLabel("Reset");
        stepButton.enable();
        clearTapeButton.enable();
        halted = flag;
    }

    public synchronized boolean action(Event event, Object obj)
    {
        TM.clearTemporaryMessage();
        if(event.target == makeButton)
            doMakeRule();
        else
        if(event.target == runButton)
        {
            if(currentData == null)
                return true;
            if(running)
            {
                runButton.disable();
                TM.stopRunning();
            } else
            {
                TM.startRunning();
                running = true;
                clearTapeButton.disable();
                stepButton.disable();
                if(halted)
                {
                    halted = false;
                    stepButton.setLabel("Step");
                }
                runButton.setLabel("Stop");
            }
        } else
        if(event.target == stepButton)
        {
            if(currentData == null)
                return true;
            if(!running)
                if(halted)
                {
                    TM.reset();
                    stepButton.setLabel("Step");
                    halted = false;
                } else
                {
                    stepButton.disable();
                    runButton.disable();
                    clearTapeButton.disable();
                    TM.doStep();
                }
        } else
        if(event.target == clearTapeButton)
            synchronized(this)
            {
                if(currentData != null)
                    TM.clearTape();
                else
                    abortLoad();
            }
        else
        if(event.target == deleteRuleButton)
        {
            if(currentData != null)
                ruleTable.doDeleteRule();
        } else
        if(event.target == loadButton)
            doLoad();
        else
        if(event.target == saveButton)
            doSave();
        else
        if(event.target == speedChoice)
        {
            int i = speedChoice.getSelectedIndex();
            if(i != currentSpeed)
            {
                TM.setSpeed(i);
                currentSpeed = i;
            }
        } else
        {
            return super.action(event, obj);
        }
        return true;
    }

    Machine TM;
    Button runButton;
    Button stepButton;
    Button clearTapeButton;
    Button deleteRuleButton;
    Button loadButton;
    Button saveButton;
    Button makeButton;
    boolean canLoad;
    boolean canSave;
    Choice speedChoice;
    int currentSpeed;
    Vector machineData;
    MachineData currentData;
    int currentMachineNumber;
    RuleTable ruleTable;
    RuleMakerCanvas ruleMaker;
    Palette palette;
    boolean running;
    boolean halted;
    Thread timer;
    int timeToAlarm;
    MachineLoader loaders[];
    TuringMachinePanel owner;
    static final int NOFOCUS = 0;
    static final int RULETABLEFOCUS = 1;
    static final int RULEMAKERFOCUS = 2;
    static final int MACHINEFOCUS = 3;
    int currentFocus;
    int currentPaletteDisplay;
    boolean focused;
}
