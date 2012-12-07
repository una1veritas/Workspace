// Decompiled by Jad v1.5.7d. Copyright 2000 Pavel Kouznetsov.
// Jad home page: http://www.geocities.com/SiliconValley/Bridge/8617/jad.html
// Decompiler options: packimports(3) 
// Source File Name:   RuleMakerCanvas.java

package tm;

import java.awt.*;

// Referenced classes of package tmcm.xTuringMachine:
//            MachineData, MachinePanel, Palette

class RuleMakerCanvas extends Canvas
{

    RuleMakerCanvas(MachinePanel machinepanel)
    {
        width = -1;
        columnLoc = new int[11];
        setBackground(Color.white);
        state = 999;
        selectedColumn = -1;
        owner = machinepanel;
    }

    public Dimension preferredSize()
    {
        return new Dimension(400, 28);
    }

    public Dimension mimimumSize()
    {
        return new Dimension(100, 28);
    }

    public void paint(Graphics g)
    {
        if(size().width != width || size().height != height)
            setUp();
        g.setFont(font);
        g.drawRect(0, 0, width, height);
        g.drawRect(1, 1, width - 2, height - 2);
        g.drawLine(0, height - 2, width, height - 2);
        g.drawLine(width - 2, 0, width - 2, height);
        g.drawLine(columnLoc[2], 0, columnLoc[2], height);
        g.drawLine(columnLoc[4], 0, columnLoc[4], height);
        g.drawLine(columnLoc[4] - 1, 0, columnLoc[4] - 1, height);
        g.drawLine(columnLoc[6], 0, columnLoc[6], height);
        g.drawLine(columnLoc[8], 0, columnLoc[8], height);
        if(state == 999)
            return;
        String s = String.valueOf(state);
        g.drawString(s, columnLoc[1] - fm.stringWidth(s) / 2, baseLine);
        if(symbol == '*')
            s = "other";
        else
            s = String.valueOf(symbol);
        g.drawString(s, columnLoc[3] - fm.stringWidth(s) / 2, baseLine);
        if(newSymbol == '*')
            s = "same";
        else
            s = String.valueOf(newSymbol);
        g.drawString(s, columnLoc[5] - fm.stringWidth(s) / 2, baseLine);
        if(direction)
            s = "R";
        else
            s = "L";
        g.drawString(s, columnLoc[7] - fm.stringWidth(s) / 2, baseLine);
        if(newState == -1)
            s = "h";
        else
            s = String.valueOf(newState);
        g.drawString(s, columnLoc[9] - fm.stringWidth(s) / 2, baseLine);
        if(hasFocus && selectedColumn >= 0)
        {
            g.setColor(Palette.hiliteColor);
            int i = columnLoc[2 * selectedColumn] + 4;
            int j = columnLoc[2 + 2 * selectedColumn] - 3 - i;
            byte byte0 = 4;
            int k = height - 8;
            g.drawRect(i, byte0, j, k);
            g.drawRect(i + 1, byte0 + 1, j - 2, k - 2);
        }
    }

    void setUp()
    {
        width = size().width;
        height = size().height;
        for(int i = 1; i < 10; i++)
            columnLoc[i] = 2 + (i * (width - 3) + 5) / 10;

        columnLoc[10] = width - 1;
        if(font == null)
        {
            font = getFont();
            fm = getFontMetrics(font);
        }
        baseLine = ((height + fm.getAscent()) - fm.getDescent()) / 2;
    }

    synchronized void setRule(int i, char c, boolean flag)
    {
        if(!data.ruleDefined(i, c))
            owner.makeButton.setLabel("Make Rule");
        else
            owner.makeButton.setLabel("Replace");
        if(selectedColumn == 2 && c == '*' && symbol != '*')
            owner.requestFocus(2, 4);
        else
        if(selectedColumn == 2 && c != '*' && symbol == '*')
            owner.requestFocus(2, 2);
        if(state != i || symbol != c)
        {
            if(data.getNewState(i, c) == 999)
            {
                newState = i;
                newSymbol = c;
                direction = true;
            } else
            {
                newState = data.getNewState(i, c);
                newSymbol = data.getNewSymbol(i, c);
                direction = data.getDirection(i, c);
            }
            state = i;
            symbol = c;
        }
        if(flag && !hasFocus)
        {
            selectedColumn = 2;
            if(c == '*')
                owner.requestFocus(2, 4);
            else
                owner.requestFocus(2, 2);
        }
        repaint(2, 2, width - 4, height - 4);
    }

    void setMachineData(MachineData machinedata)
    {
        data = machinedata;
        if(machinedata == null)
        {
            state = 999;
            selectedColumn = -1;
            owner.makeButton.disable();
            repaint();
            return;
        } else
        {
            setRule(0, '_', false);
            owner.makeButton.enable();
            selectedColumn = 2;
            repaint();
            return;
        }
    }

    void ruleMade()
    {
        int i = "_$01xyz*".indexOf(symbol);
        i++;
        int j = state;
        if(i >= 8)
        {
            i = 0;
            if(++j >= 25)
                j = 0;
        }
        char c = "_$01xyz*".charAt(i);
        setRule(j, c, false);
    }

    final void repaintContents()
    {
        repaint(2, 2, width - 4, height - 4);
    }

    final void repaintColumn(int i)
    {
        int j = columnLoc[2 * i] + 2;
        int k = columnLoc[2 + 2 * i] - 1 - j;
        repaint(j, 2, k, height - 4);
    }

    void focusIsOn()
    {
        hasFocus = true;
        firstKeyDownInSelection = true;
        repaintContents();
    }

    void focusIsOff()
    {
        hasFocus = false;
        repaintContents();
    }

    void processKey(int i)
    {
        if(i >= 32 && i <= 127)
        {
            if(i == 32)
                i = 35;
            i = Character.toLowerCase((char)i);
            switch(selectedColumn)
            {
            case 0: // '\0'
                if(i >= 48 && i <= 57)
                {
                    state = (10 * state + i) - 48;
                    if(state >= 25 || firstKeyDownInSelection)
                        state = i - 48;
                    repaintColumn(0);
                    if(data.ruleDefined(state, symbol))
                        owner.makeButton.setLabel("Replace");
                    else
                        owner.makeButton.setLabel("Make Rule");
                    firstKeyDownInSelection = false;
                }
                break;

            case 1: // '\001'
                if(i == 105)
                    i = 49;
                else
                if(i == 111)
                    i = 48;
                if("_$01xyz*".indexOf((char)i) >= 0)
                {
                    if(symbol == '*' && (char)i != '*' && newSymbol == '*')
                    {
                        newSymbol = (char)i;
                        repaintColumn(2);
                    }
                    symbol = (char)i;
                    repaintColumn(1);
                    if(data.ruleDefined(state, symbol))
                        owner.makeButton.setLabel("Replace");
                    else
                        owner.makeButton.setLabel("Make Rule");
                }
                break;

            case 2: // '\002'
                if(i == 105)
                    i = 49;
                else
                if(i == 111)
                    i = 48;
                if("_$01xyz*".indexOf((char)i) >= 0 && (i != 42 || symbol == '*'))
                {
                    newSymbol = (char)i;
                    repaintColumn(2);
                }
                break;

            case 3: // '\003'
                if(i == 108)
                {
                    direction = false;
                    repaintColumn(3);
                } else
                if(i == 114)
                {
                    direction = true;
                    repaintColumn(3);
                }
                break;

            case 4: // '\004'
                if(i == 104)
                {
                    newState = -1;
                    repaintColumn(4);
                } else
                if(i >= 48 && i <= 57)
                {
                    int j;
                    if(newState == -1)
                    {
                        j = i - 48;
                    } else
                    {
                        j = (10 * newState + i) - 48;
                        if(j >= 25 || firstKeyDownInSelection)
                            j = i - 48;
                    }
                    newState = j;
                    repaintColumn(4);
                    firstKeyDownInSelection = false;
                }
                break;
            }
        } else
        if(i == 1006 || i == 1007)
        {
            if(i == 1006)
                selectedColumn = selectedColumn != 0 ? selectedColumn - 1 : 4;
            else
                selectedColumn = selectedColumn != 4 ? selectedColumn + 1 : 0;
            if(selectedColumn == 0)
                owner.requestFocus(2, 1);
            else
            if(selectedColumn == 1)
                owner.requestFocus(2, 4);
            else
            if(selectedColumn == 2)
            {
                if(symbol == '*')
                    owner.requestFocus(2, 4);
                else
                    owner.requestFocus(2, 2);
            } else
            if(selectedColumn == 3)
                owner.requestFocus(2, 3);
            else
            if(selectedColumn == 4)
                owner.requestFocus(2, 5);
        } else
        if(i == 1004 || i == 1005)
        {
            int k = "_$01xyz*".indexOf(symbol);
            if(i == 1004)
            {
                if(--k < 0)
                {
                    k = 7;
                    state--;
                    if(state < 0)
                        state = 24;
                }
            } else
            if(++k >= 8)
            {
                k = 0;
                state++;
                if(state >= 25)
                    state = 0;
            }
            char c = "_$01xyz*".charAt(k);
            if(selectedColumn == 2 && c == '*' && symbol != '*')
                owner.requestFocus(2, 4);
            else
            if(selectedColumn == 2 && c != '*' && symbol == '*')
                owner.requestFocus(2, 2);
            symbol = c;
            if(!data.ruleDefined(state, symbol))
                owner.makeButton.setLabel("Make Rule");
            else
                owner.makeButton.setLabel("Replace");
            if(data.getNewState(state, symbol) == 999)
            {
                newState = state;
                newSymbol = symbol;
                direction = true;
            } else
            {
                newState = data.getNewState(state, symbol);
                newSymbol = data.getNewSymbol(state, symbol);
                direction = data.getDirection(state, symbol);
            }
            repaintContents();
        } else
        if(i == 13 || i == 10)
            owner.doMakeRule();
    }

    void processPaletteClick(int i, int j)
    {
        firstKeyDownInSelection = true;
        if(j == 3)
            processKey(i != 0 ? 82 : 76);
        else
        if(j == 2 || j == 4)
            processKey("_$01xyz*".charAt(i));
        else
        if(j == 1 || j == 5)
            if(i == -1)
                processKey(104);
            else
            if(selectedColumn == 0)
            {
                state = i;
                if(!data.ruleDefined(state, symbol))
                    owner.makeButton.setLabel("Make Rule");
                else
                    owner.makeButton.setLabel("Replace");
                repaintColumn(0);
            } else
            if(selectedColumn == 4)
            {
                newState = i;
                repaintColumn(4);
            }
    }

    public synchronized boolean mouseDown(Event event, int i, int j)
    {
        if(data == null)
            return true;
        int k = i / (width / 5);
        if(k < 0)
            k = 0;
        else
        if(k > 4)
            k = 4;
        if(hasFocus && k == selectedColumn)
            return true;
        selectedColumn = k;
        byte byte0;
        if(k == 0)
            byte0 = 1;
        else
        if(k == 1)
            byte0 = 4;
        else
        if(k == 2)
        {
            if(symbol == '*')
                byte0 = 4;
            else
                byte0 = 2;
        } else
        if(k == 3)
            byte0 = 3;
        else
            byte0 = 5;
        owner.requestFocus(2, byte0);
        return true;
    }

    int width;
    int height;
    int baseLine;
    int columnLoc[];
    Font font;
    FontMetrics fm;
    int state;
    char symbol;
    int newState;
    char newSymbol;
    boolean direction;
    int selectedColumn;
    boolean hasFocus;
    MachineData data;
    MachinePanel owner;
    boolean firstKeyDownInSelection;
}
