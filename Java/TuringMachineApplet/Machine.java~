// Decompiled by Jad v1.5.7d. Copyright 2000 Pavel Kouznetsov.
// Jad home page: http://www.geocities.com/SiliconValley/Bridge/8617/jad.html
// Decompiler options: packimports(3) 
// Source File Name:   Machine.java

package tm;

import java.awt.*;

// Referenced classes of package tmcm.xTuringMachine:
//            MachineData, MachinePanel, Palette, RuleMakerCanvas

class Machine extends Panel
    implements Runnable
{

    Machine(MachinePanel machinepanel)
    {
        changedRect = new Rectangle();
        selectedItem = 0x3b9aca00;
        width = -1;
        message = new String[3];
        status = 0;
        setBackground(backgroundColor);
        owner = machinepanel;
    }

    public Dimension preferredSize()
    {
        return new Dimension(500, 76);
    }

    public Dimension minimumSize()
    {
        return new Dimension(100, 76);
    }

    synchronized void setMachineData(MachineData machinedata, int i)
    {
        if(status == 3)
            stopRunning();
        data = machinedata;
        centerSquare = i;
        machineState = 0;
        currentSquare = i;
        selectedItem = 0x3b9aca00;
        status = 0;
        changed = true;
        clearMessage();
        setChangedAll();
        repaint();
    }

    synchronized void setMessage(String s, String s1, String s2)
    {
        stopRunning();
        message[0] = s;
        message[1] = s1;
        message[2] = s2;
        temporaryMessage = false;
        setStatus(5);
        repaint();
    }

    synchronized void clearMessage()
    {
        if(status == 5)
        {
            message[0] = message[1] = message[2] = null;
            temporaryMessage = false;
            setStatus(0);
            repaint();
        }
    }

    synchronized void clearTemporaryMessage()
    {
        if(temporaryMessage && status == 5)
        {
            message[0] = message[1] = message[2] = null;
            temporaryMessage = false;
            setStatus(0);
            repaint();
        }
    }

    synchronized void setTemporaryMessage(String s, String s1, String s2)
    {
        stopRunning();
        selectedItem = 0x3b9aca00;
        owner.dropFocus(3);
        message[0] = s;
        message[1] = s1;
        message[2] = s2;
        temporaryMessage = true;
        setStatus(5);
        repaint();
    }

    public synchronized void paint(Graphics g)
    {
        if(tapeFont == null || width != size().width || height != size().height)
            setUp();
        if(status == 5)
        {
            g.setColor(backgroundColor);
            g.fillRect(0, 0, size().width, size().height);
            g.setColor(tapeColor);
            g.drawString(message[0], 10, 10 + tapeFM.getAscent());
            if(message[1] != null)
                g.drawString(message[1], 10, 10 + tapeFM.getAscent() + tapeFM.getHeight());
            if(message[2] != null)
                g.drawString(message[2], 10, 10 + tapeFM.getAscent() + 2 * tapeFM.getHeight());
            return;
        }
        if(OSC == null)
        {
            setChangedAll();
            doDraw(g);
        } else
        {
            if(changed)
                doDraw(OSG);
            g.drawImage(OSC, 0, 0, this);
        }
    }

    public void update(Graphics g)
    {
        paint(g);
    }

    void setUp()
    {
        width = size().width;
        height = size().height;
        tapeFont = getFont();
        tapeFM = getFontMetrics(tapeFont);
        charWidth = tapeFM.charWidth('9');
        squareSize = tapeFM.getAscent() + tapeFM.getDescent() + 10;
        tapeTop = height - squareSize - 3;
        baseLine = tapeTop + ((squareSize + tapeFM.getAscent()) - tapeFM.getDescent()) / 2;
        machineHeight = tapeTop - 12;
        machineFont = new Font(tapeFont.getName(), 0, 24);
        machineFM = getFontMetrics(machineFont);
        if(machineFM.getAscent() + machineFM.getDescent() + 14 > machineHeight)
        {
            machineFont = new Font(tapeFont.getName(), 0, 18);
            machineFM = getFontMetrics(machineFont);
            if(machineFM.getAscent() + machineFM.getDescent() + 14 > machineHeight)
            {
                machineFont = new Font(tapeFont.getName(), 0, 14);
                machineFM = getFontMetrics(machineFont);
                if(machineFM.getAscent() + machineFM.getDescent() + 14 > machineHeight)
                {
                    machineFont = new Font(tapeFont.getName(), 0, 12);
                    machineFM = getFontMetrics(machineFont);
                }
            }
        }
        machineWidth = machineFM.stringWidth("99") + 18;
        if(machineWidth < (4 * machineHeight) / 5)
            machineWidth = (4 * machineHeight) / 5;
        machineBaseline = 6 + ((machineHeight + machineFM.getAscent()) - machineFM.getDescent()) / 2;
        try
        {
            OSC = createImage(width, height);
            OSG = OSC.getGraphics();
        }
        catch(OutOfMemoryError _ex)
        {
            OSC = null;
            OSG = null;
        }
        setChangedAll();
    }

    void doDraw(Graphics g) {
        g.setColor(backgroundColor);
        g.fillRect(changedRect.x, changedRect.y, changedRect.width, changedRect.height);
        g.setColor(tapeColor);
        g.drawLine(0, tapeTop, width, tapeTop);
        g.drawLine(0, height - 3, width, height - 3);
        if(changedRect.y + changedRect.height > tapeTop) {
            // drawing tape.
            int i = centerSquare - (width / 2 - changedRect.x) / squareSize - 1;
            int l = centerSquare + ((changedRect.x + changedRect.width) - width / 2) / squareSize + 1;
            int k1 = width / 2 - (centerSquare - i) * squareSize - (squareSize + 1) / 2;
            g.drawLine(k1, tapeTop, k1, height - 4);
            g.setFont(tapeFont);
            for(int j2 = i; j2 <= l; j2++) {
                char c = data != null ? data.getTape(j2) : '_';
                if(blink && j2 == currentSquare)
                    g.fillRect(k1, tapeTop, squareSize, squareSize);
                if(c != '_') {
                    g.setColor(symbolColor);
                    g.drawString(String.valueOf(c), k1 + (squareSize - charWidth) / 2, baseLine);
                    g.setColor(tapeColor);
                }
                k1 += squareSize;
                g.drawLine(k1, tapeTop, k1, height - 4);
            }

            if(focused && selectedItem != 0x3b9aca00 && selectedItem != 0xc4653600)  {
                g.setColor(Palette.hiliteColor);
                int l1 = (width - squareSize) / 2 - (centerSquare - selectedItem) * squareSize;
                g.drawRect(l1, tapeTop, squareSize + 1, squareSize + 1);
                g.drawRect(l1 + 1, tapeTop + 1, squareSize - 1, squareSize - 1);
            }
        }
        if(status == 3 && speed == 0)
        {
            g.setColor(machineColor);
            int j = width / 2 - (centerSquare - currentSquare) * squareSize;
            g.fillRect((j - squareSize / 2) + 1, tapeTop - 3, squareSize - 1, 3);
        } else
        if(changedRect.y < tapeTop) {
            // drawing machine. 
            int k = (width - machineWidth) / 2 - (centerSquare - currentSquare) * squareSize;
            int i1 = width / 2 - (centerSquare - currentSquare) * squareSize;
            if(changedRect.x <= k + machineWidth || changedRect.x + changedRect.width >= k) {
                // drawing the outline of the machine.
                g.setColor(machineColor); 
                int xseq[] = {k, k+machineWidth, k+machineWidth, k+machineWidth/2, k};
                int yseq[] = {6, 6, machineHeight, tapeTop -3, machineHeight};
                int xinseq[] = {k+2, k+machineWidth-2, k+machineWidth-2, k+machineWidth/2, k+2};
                int yinseq[] = {6+2, 6+2, machineHeight-2, tapeTop -3 -2, machineHeight -2};
                g.fillPolygon(xseq, yseq, 5);
                
                //g.fillRect((i1 - squareSize / 2) + 1, tapeTop - 3, squareSize - 1, 3);
                //g.fillRect(i1 - 2, tapeTop - 6, 4, 4);
                //g.fillRoundRect(k, 6, machineWidth, machineHeight, 12, 12);
                
                //drawing internal region of machine;
                if(focused && selectedItem == 0xc4653600) {
                    // when grabbed
                    g.setColor(Palette.hiliteColor);
                    //g.fillRoundRect(k + 3, 9, machineWidth - 6, machineHeight - 6, 12, 12);
                    g.fillPolygon(xseq, yseq, 5);
                    // g.setColor(insideMachineColor);
                    // g.fillRoundRect(k + 5, 11, machineWidth - 10, machineHeight - 10, 12, 12);
                }
                g.setColor(insideMachineColor);
                //g.fillRoundRect(k + 3, 9, machineWidth - 6, machineHeight - 6, 12, 12);
                g.fillPolygon(xinseq, yinseq, 5);
                String s1 = machineState != -1 ? String.valueOf(machineState) : "h";
                g.setColor(stateColor);
                g.setFont(machineFont);
                g.drawString(s1, i1 - machineFM.stringWidth(s1) / 2, machineBaseline);
            }
        }
        if(status == 1)
        {
            String s = "No Rule Defined!";
            int j1 = machineFM.stringWidth(s);
            if(j1 + 10 <= width - machineWidth / 2)
            {
                g.setFont(machineFont);
            } else
            {
                g.setFont(tapeFont);
                j1 = tapeFM.stringWidth(s);
            }
            int i2;
            if(currentSquare <= centerSquare)
                i2 = ((width + machineWidth) / 2 - (centerSquare - currentSquare) * squareSize) + 10;
            else
                i2 = (width - machineWidth) / 2 - (centerSquare - currentSquare) * squareSize - j1 - 10;
            g.setColor(stateColor);
            g.drawString(s, i2, machineBaseline);
        }
        changed = false;
        changedRect.reshape(0, 0, 0, 0);
    }

    synchronized void setBlinking(boolean flag)
    {
        blink = flag;
    }

    synchronized void setChangedAll()
    {
        changedRect.reshape(0, 0, width, height);
        changed = true;
    }

    synchronized void setChanged(int i, int j, int k, int l)
    {
        if(changedRect.isEmpty())
        {
            changedRect.reshape(i, j, k, l);
        } else
        {
            changedRect.add(i, j);
            changedRect.add(i + k, j + l);
        }
        changed = true;
    }

    synchronized void setStatus(int i)
    {
        status = i;
        notify();
    }

    synchronized int getStatus()
    {
        return status;
    }

    synchronized void focusIsOn()
    {
        focused = true;
        repaintSelection();
        firstKeyDownInSelection = true;
    }

    synchronized void focusIsOff()
    {
        focused = false;
        repaintSelection();
    }

    synchronized void processKey(int i)
    {
        if(i >= 32 && i <= 127)
        {
            if(i == 32)
                i = 35;
            else
                i = Character.toLowerCase((char)i);
            if(selectedItem == 0xc4653600)
            {
                if(i == 104)
                {
                    if(machineState != -1)
                        owner.stepButton.setLabel("Reset");
                    machineState = -1;
                    repaintSelection();
                } else
                if(i <= 57 && i >= 48)
                {
                    if(machineState == -1)
                        owner.stepButton.setLabel("Step");
                    if(machineState == -1 || firstKeyDownInSelection)
                        machineState = i - 48;
                    else
                        machineState = (10 * machineState + i) - 48;
                    if(machineState >= 25)
                        machineState = i - 48;
                    repaintSelection();
                    firstKeyDownInSelection = false;
                }
            } else
            if(selectedItem != 0x3b9aca00)
            {
                if(i == 105)
                    i = 49;
                else
                if(i == 111)
                    i = 48;
                if(i != 42 && "_$01xyz*".indexOf((char)i) >= 0 && data != null)
                {
                    data.setTape(selectedItem, (char)i);
                    selectItem(selectedItem + 1);
                }
            }
        } else
        if(i == 1006)
        {
            if(Math.abs(selectedItem) < 0x3b9aca00)
                selectItem(selectedItem - 1);
        } else
        if(i == 1007)
        {
            if(Math.abs(selectedItem) < 0x3b9aca00)
                selectItem(selectedItem + 1);
        } else
        if(i == 1004 || i == 1005)
        {
            if(selectedItem == 0xc4653600)
                selectItem(currentSquare);
            else
                selectItem(0xc4653600);
        } else
        if(i == 1000)
        {
            if(data != null)
                selectItem(data.firstFilledSquare());
        } else
        if(i == 1001 && data != null)
            selectItem(data.lastFilledSquare());
    }

    synchronized void processPaletteClick(int i, int j)
    {
        firstKeyDownInSelection = true;
        if(j == 2)
            processKey("_$01xyz*".charAt(i));
        else
        if(j == 5 && selectedItem == 0xc4653600)
        {
            int k;
            if(i == -1)
            {
                if(machineState != -1)
                    owner.stepButton.setLabel("Reset");
                k = -1;
            } else
            {
                if(machineState == -1)
                    owner.stepButton.setLabel("Step");
                k = i;
            }
            if(k == machineState)
                return;
            machineState = k;
            repaintSelection();
        }
    }

    synchronized void selectItem(int i)
    {
        if(status == 1)
        {
            setStatus(0);
            setChanged(0, 0, width, tapeTop);
            repaint(0, 0, width, tapeTop);
        }
        if(i != selectedItem && focused)
        {
            repaintSelection();
            firstKeyDownInSelection = true;
        }
        selectedItem = i;
        if(i == 0xc4653600)
            owner.requestFocus(3, 5);
        else
        if(i != 0x3b9aca00)
            owner.requestFocus(3, 2);
    }

    void repaintSelection()
    {
        if(selectedItem == 0xc4653600)
        {
            int i = (width - squareSize) / 2 - (centerSquare - currentSquare) * squareSize;
            if(i < 0 || i + squareSize > width)
            {
                showSquare(currentSquare);
            } else
            {
                int k = (width - machineWidth) / 2 - (centerSquare - currentSquare) * squareSize - 2;
                setChanged(k, 3, k + machineWidth + 10, tapeTop - 3);
                repaint(k, 3, k + machineWidth + 10, tapeTop - 3);
            }
        } else
        if(selectedItem != 0x3b9aca00)
        {
            int j = (width - squareSize) / 2 - (centerSquare - selectedItem) * squareSize;
            if(j < 0 || j + squareSize > width)
            {
                showSquare(selectedItem);
            } else
            {
                setChanged(j - 1, tapeTop, squareSize + 3, squareSize + 2);
                repaint(j - 1, tapeTop, squareSize + 3, squareSize + 2);
            }
        }
    }

    synchronized void showSquare(int i)
    {
        int j = (width - squareSize) / 2 - (centerSquare - i) * squareSize;
        if(j < 0)
        {
            centerSquare = i + width / squareSize / 4;
            setChangedAll();
            repaint();
        } else
        if(j + squareSize > width)
        {
            centerSquare = i - width / squareSize / 4;
            setChangedAll();
            repaint();
        }
    }

    public boolean mouseDown(Event event, int i, int j)
    {
        if(temporaryMessage && message[0] != null)
        {
            clearTemporaryMessage();
            return true;
        }
        if(data == null)
            return true;
        if(getStatus() > 1)
        {
            dragging = false;
            return true;
        }
        if(getStatus() == 1)
        {
            setStatus(0);
            setChanged(0, 0, width, tapeTop);
            repaint();
        }
        start_x = i;
        start_y = j;
        startCurrentSquare = currentSquare;
        startCenterSquare = centerSquare;
        shiftedClick = event.metaDown() || event.controlDown();
        int k = (width - machineWidth) / 2 - (centerSquare - currentSquare) * squareSize;
        if(!shiftedClick)
            if(j >= tapeTop)
            {
                int l = i - width / 2;
                if(l > 0)
                    selectItem(centerSquare + (l + squareSize / 2) / squareSize);
                else
                    selectItem(centerSquare - (-l + squareSize / 2) / squareSize);
            } else
            if(i > k - 5 && i < k + machineWidth + 5)
                selectItem(0xc4653600);
        dragging = shiftedClick || j >= tapeTop || i > k - 5 && i < k + machineWidth + 5;
        return true;
    }

    public boolean mouseDrag(Event event, int i, int j)
    {
        if(!dragging)
            return true;
        int k = i - start_x;
        if(k > 0)
            k = (k + squareSize / 2) / squareSize;
        else
            k = -((-k + squareSize / 2) / squareSize);
        if(shiftedClick)
            centerSquare = startCenterSquare - k;
        else
        if(start_y >= tapeTop)
        {
            centerSquare = startCenterSquare - k;
            currentSquare = startCurrentSquare - k;
        } else
        {
            currentSquare = startCurrentSquare + k;
        }
        setChangedAll();
        repaint();
        return true;
    }

    synchronized void setSpeed(int i)
    {
        if(status == 3 && (i == 0 || speed == 0))
        {
            setChanged(0, 0, width, tapeTop);
            repaint();
        }
        speed = i;
        notify();
    }

    synchronized int getSpeed()
    {
        return speed;
    }

    synchronized void startRunning()
    {
        if(status == 1)
        {
            setChanged(0, 0, width, tapeTop);
            setStatus(0);
            repaint();
            try
            {
                wait(100L);
            }
            catch(InterruptedException _ex) { }
        }
        if(data == null || status == 5)
        {
            owner.doneRunning(false);
            return;
        }
        owner.dropFocus(3);
        status = 3;
        if(speed == 0)
        {
            setChanged(0, 0, width, tapeTop);
            repaint(0, 0, width, tapeTop);
        }
        if(runner == null || !runner.isAlive())
        {
            runner = new Thread(this);
            runner.start();
        } else
        {
            notify();
        }
    }

    synchronized void doStep()
    {
        if(data == null || status == 5)
        {
            owner.doneStep(false);
            return;
        }
        if(status == 1)
        {
            setChanged(0, 0, width, tapeTop);
            repaint();
            setStatus(0);
            try
            {
                wait(100L);
            }
            catch(InterruptedException _ex) { }
        }
        owner.dropFocus(3);
        status = 4;
        if(runner == null || !runner.isAlive())
        {
            runner = new Thread(this);
            runner.start();
        } else
        {
            notify();
        }
    }

    synchronized void stopRunning()
    {
        if(status != 3)
            return;
        setStatus(2);
        try
        {
            wait(1000L);
        }
        catch(InterruptedException _ex) { }
        setStatus(0);
        setChangedAll();
        repaint();
        owner.doneRunning(machineState == -1);
    }

    synchronized void reset()
    {
        if(status == 1)
        {
            setChanged(0, 0, width, tapeTop);
            repaint();
            setStatus(0);
        }
        machineState = 0;
        setChanged(0, 0, width, tapeTop);
        repaint();
    }

    synchronized void clearTape()
    {
        if(data != null)
            data.clearTape();
        setChangedAll();
        currentSquare = 0;
        centerSquare = 0;
        if(status == 1)
            setStatus(0);
        if(selectedItem != 0x3b9aca00 && selectedItem != 0xc4653600)
            selectedItem = 0;
        repaint();
    }

    synchronized void doWait(int i)
    {
        try
        {
            wait(i);
        }
        catch(InterruptedException _ex) { }
    }

    void executeOneStep()
    {
        if(machineState == -1)
        {
            machineState = 0;
            setChanged(0, 0, width, tapeTop);
            repaint();
            doWait(100);
        }
        char c = data.getTape(currentSquare);
        int i = data.getNewState(machineState, c);
        if(i == 999)
        {
            setStatus(1);
            setChangedAll();
            owner.ruleMaker.setRule(machineState, c, true);
            int j = (width - squareSize) / 2 - (centerSquare - currentSquare) * squareSize;
            if(j < 0 || j + squareSize > width)
                showSquare(currentSquare);
            else
                repaint();
            return;
        }
        char c2 = data.getNewSymbol(machineState, c);
        boolean flag = data.getDirection(machineState, c);
        int k = (width - squareSize) / 2 - (centerSquare - currentSquare) * squareSize;
        if(getSpeed() > 1)
        {
            setBlinking(true);
            setChanged(k, tapeTop, squareSize, squareSize);
            repaint();
            if(getSpeed() == 2)
                doWait(100);
            else
                doWait(200);
            setBlinking(false);
            data.setTape(currentSquare, c2);
            setChanged(k, tapeTop, squareSize, squareSize);
            if(getSpeed() > 2)
            {
                repaint();
                doWait(100);
            }
        } else
        if(c2 != c)
        {
            data.setTape(currentSquare, c2);
            setChanged(k, tapeTop, squareSize, squareSize);
        }
        if(flag)
        {
            currentSquare++;
            setChanged((k + (squareSize - machineWidth) / 2) - 2, 0, machineWidth * 2 + 4, tapeTop);
        } else
        {
            currentSquare--;
            setChanged((k + (squareSize - 3 * machineWidth) / 2) - 2, 0, machineWidth * 2 + 4, tapeTop);
        }
        machineState = i;
        if(machineState != -1)
        {
            char c1 = data.getTape(currentSquare);
            if(data.getNewState(machineState, c1) == 999)
            {
                setStatus(1);
                setChangedAll();
                repaint();
                owner.ruleMaker.setRule(machineState, c1, true);
            } else
            if(getSpeed() > 1)
                if(data.ruleDefined(machineState, c1))
                    owner.ruleMaker.setRule(machineState, c1, false);
                else
                    owner.ruleMaker.setRule(machineState, '*', false);
        }
        k = (width - squareSize) / 2 - (centerSquare - currentSquare) * squareSize;
        if(k < 0 || k + squareSize > width)
            showSquare(currentSquare);
        else
            repaint();
    }

    public void run()
    {
        do
        {
            int i;
            synchronized(this)
            {
                for(i = getStatus(); i != 3 && i != 4 && i != 2; i = getStatus())
                    try
                    {
                        wait();
                    }
                    catch(InterruptedException _ex) { }

            }
            if(i != 2)
                executeOneStep();
            synchronized(this)
            {
                if(i == 4)
                {
                    if(status != 1)
                        setStatus(0);
                    owner.doneStep(machineState == -1);
                } else
                if(machineState == -1)
                {
                    setStatus(0);
                    if(getSpeed() == 0)
                    {
                        setChanged(0, 0, width, tapeTop);
                        repaint(0, 0, width, tapeTop);
                    }
                    owner.doneRunning(true);
                } else
                if(getStatus() == 2)
                    notify();
                else
                if(getStatus() == 1)
                    owner.doneRunning(false);
                else
                    try
                    {
                        wait(speedDelay[speed]);
                    }
                    catch(InterruptedException _ex) { }
            }
        } while(true);
    }

    static final Color backgroundColor = new Color(255, 240, 220);
    static final Color tapeColor;
    static final Color machineColor;
    static final Color insideMachineColor;
    static final Color symbolColor;
    static final Color stateColor;
    int squareSize;
    int machineWidth;
    int machineHeight;
    Font tapeFont;
    int charWidth;
    int baseLine;
    int machineBaseline;
    int tapeTop;
    Font machineFont;
    FontMetrics machineFM;
    FontMetrics tapeFM;
    Rectangle changedRect;
    static final int NOSELECTION = 0x3b9aca00;
    static final int MACHINESELECTED = 0xc4653600;
    int selectedItem;
    int width;
    int height;
    Image OSC;
    Graphics OSG;
    MachineData data;
    String message[];
    boolean temporaryMessage;
    int centerSquare;
    double squaresVisible;
    int currentSquare;
    int machineState;
    int speed;
    int speedDelay[] = {
        20, 100, 300, 500, 1500
    };
    boolean blink;
    Thread runner;
    static final int IDLE = 0;
    static final int NORULE = 1;
    static final int STOPPING = 2;
    static final int RUNNING = 3;
    static final int STEPPING = 4;
    static final int MESSAGEDISPLAY = 5;
    private int status;
    MachinePanel owner;
    boolean focused;
    boolean changed;
    boolean firstKeyDownInSelection;
    int start_x;
    int start_y;
    int startCurrentSquare;
    int startCenterSquare;
    boolean shiftedClick;
    boolean dragging;

    static 
    {
        tapeColor = new Color(60, 50, 40);
        machineColor = tapeColor;
        insideMachineColor = Color.white;
        symbolColor = Color.blue;
        stateColor = Color.red;
    }
}
