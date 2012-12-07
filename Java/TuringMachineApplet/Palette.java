// Decompiled by Jad v1.5.7d. Copyright 2000 Pavel Kouznetsov.
// Jad home page: http://www.geocities.com/SiliconValley/Bridge/8617/jad.html
// Decompiler options: packimports(3) 
// Source File Name:   Palette.java

//package tm;

import java.awt.*;

// Referenced classes of package tmcm.xTuringMachine:
//            MachinePanel

class Palette extends Canvas
{

    Palette(MachinePanel machinepanel)
    {
        display = 0;
        width = -1;
        height = -1;
        selectedItem = -1;
        selectRect = new Rectangle();
        owner = machinepanel;
        setBackground(backgroundColor);
    }

    public Dimension preferredSize()
    {
        return new Dimension(400, 28);
    }

    public Dimension mimimumSize()
    {
        return new Dimension(100, 24);
    }

    public void paint(Graphics g)
    {
        if(size().width != width || size().height != height)
            setUp();
        g.setColor(textColor);
        g.drawRect(0, 0, width, height);
        g.drawRect(1, 1, width - 2, height - 2);
        g.drawLine(width - 2, 0, width - 2, height);
        g.drawLine(0, height - 2, width, height - 2);
        g.setFont(font);
        if(inSelection)
        {
            g.setColor(hitColor);
            g.fillRect(selectRect.x, selectRect.y, selectRect.width, selectRect.height);
            g.setColor(textColor);
        }
        switch(display)
        {
        case 0: // '\0'
            return;

        case 5: // '\005'
            leftOffset = fm.stringWidth("States:") + 16;
            g.drawString("States:", 9, baseLine);
            g.setColor(itemColor);
            int i = fm.stringWidth("00") + 6;
            if(leftOffset + 26 * i >= width)
            {
                g.setFont(smallFont);
                i = smallFm.stringWidth("00") + 6;
            }
            if(leftOffset + 26 * i < width)
            {
                twoRowsOfStates = false;
                itemWidth = (width - leftOffset) / 26;
                g.drawString("h", leftOffset + (itemWidth - fm.stringWidth("h")) / 2, baseLine);
                for(int i1 = 0; i1 < 25; i1++)
                {
                    String s = String.valueOf(i1);
                    g.drawString(s, leftOffset + (i1 + 1) * itemWidth + (itemWidth - fm.stringWidth(s)) / 2, baseLine);
                }

            } else
            {
                twoRowsOfStates = true;
                itemWidth = (width - leftOffset) / 13;
                int j1 = height / 2 - 1;
                int k2 = height - 4;
                g.drawString("h", leftOffset + (itemWidth - smallFm.stringWidth("h")) / 2, j1);
                boolean flag = false;
                for(int i3 = 0; i3 < 25; i3++)
                {
                    String s4 = String.valueOf(i3);
                    int k3 = flag ? j1 : k2;
                    g.drawString(s4, leftOffset + ((i3 + 1) / 2) * itemWidth + (itemWidth - fm.stringWidth(s4)) / 2, k3);
                    flag ^= true;
                }

            }
            return;

        case 1: // '\001'
            leftOffset = fm.stringWidth("States:") + 16;
            g.drawString("States:", 9, baseLine);
            g.setColor(itemColor);
            int j = fm.stringWidth("00") + 6;
            if(leftOffset + 25 * j >= width)
            {
                g.setFont(smallFont);
                j = smallFm.stringWidth("00") + 6;
            }
            if(leftOffset + 25 * j < width)
            {
                twoRowsOfStates = false;
                itemWidth = (width - leftOffset) / 25;
                for(int k1 = 0; k1 < 25; k1++)
                {
                    String s1 = String.valueOf(k1);
                    g.drawString(s1, leftOffset + k1 * itemWidth + (itemWidth - fm.stringWidth(s1)) / 2, baseLine);
                }

            } else
            {
                twoRowsOfStates = true;
                itemWidth = (width - leftOffset) / 13;
                int l1 = height / 2 - 1;
                int l2 = height - 4;
                boolean flag1 = true;
                for(int j3 = 0; j3 < 25; j3++)
                {
                    String s5 = String.valueOf(j3);
                    int l3 = flag1 ? l1 : l2;
                    g.drawString(s5, leftOffset + (j3 / 2) * itemWidth + (itemWidth - fm.stringWidth(s5)) / 2, l3);
                    flag1 ^= true;
                }

            }
            return;

        case 4: // '\004'
            leftOffset = fm.stringWidth("Symbols:") + 16;
            g.drawString("Symbols:", 9, baseLine);
            int k = width - smallFm.stringWidth("(* = default)") - 5;
            itemWidth = (k - leftOffset - 5) / 8;
            g.setColor(itemColor);
            for(int i2 = 0; i2 < "_$01xyz*".length(); i2++)
            {
                String s2 = String.valueOf("_$01xyz*".charAt(i2));
                g.drawString(s2, (leftOffset + i2 * itemWidth + itemWidth / 2) - 3, baseLine);
            }

            g.setFont(smallFont);
            g.setColor(textColor);
            g.drawString("(_ = blank)", k, height / 2 - 1);
            g.drawString("(* = default)", k, height - 4);
            return;

        case 2: // '\002'
            leftOffset = fm.stringWidth("Symbols:") + 16;
            g.drawString("Symbols:", 9, baseLine);
            int l = width - fm.stringWidth("(* = blank)") - 5;
            itemWidth = (l - leftOffset - 5) / 7;
            g.setColor(itemColor);
            for(int j2 = 0; j2 < "_$01xyz*".length() - 1; j2++)
            {
                String s3 = String.valueOf("_$01xyz*".charAt(j2));
                g.drawString(s3, (leftOffset + j2 * itemWidth + itemWidth / 2) - 3, baseLine);
            }

            g.setColor(textColor);
            g.drawString("(_ = blank)", l, baseLine);
            return;

        case 3: // '\003'
            leftOffset = fm.stringWidth("Directions:") + 16;
            g.drawString("Directions:", 9, baseLine);
            itemWidth = 40;
            g.setColor(itemColor);
            g.drawString("L", leftOffset + 17, baseLine);
            g.drawString("R", leftOffset + 57, baseLine);
            return;
        }
    }

    void setUp()
    {
        width = size().width;
        height = size().height;
        font = getFont();
        fm = getFontMetrics(font);
        font.getSize();
        smallFont = new Font(font.getName(), 0, 9);
        smallFm = getFontMetrics(smallFont);
        baseLine = ((height + fm.getAscent()) - fm.getDescent()) / 2;
    }

    public void reshape(int i, int j, int k, int l)
    {
        super.reshape(i, j, k, l);
        inSelection = false;
        selectedItem = -1;
    }

    void setDisplay(int i)
    {
        if(display == i && selectedItem == -1)
        {
            return;
        } else
        {
            display = i;
            inSelection = false;
            selectedItem = -1;
            repaint();
            return;
        }
    }

    void checkPoint(int i, int j, boolean flag)
    {
        inSelection = false;
        if(display == 0 || i < leftOffset)
            return;
        int k = (i - leftOffset) / itemWidth;
        switch(display)
        {
        default:
            break;

        case 1: // '\001'
        case 5: // '\005'
            if(twoRowsOfStates)
            {
                k = 2 * k;
                if(j > height / 2)
                    k++;
            }
            if(k >= (display != 1 ? 26 : 25))
                return;
            if(flag)
            {
                selectedItem = k;
                if(twoRowsOfStates)
                {
                    selectRect.x = leftOffset + (k / 2) * itemWidth;
                    selectRect.width = itemWidth;
                    selectRect.y = (k & 0x1) != 0 ? height / 2 : 3;
                    selectRect.height = height / 2 - 3;
                } else
                {
                    selectRect.x = leftOffset + k * itemWidth;
                    selectRect.width = itemWidth;
                    selectRect.y = 4;
                    selectRect.height = height - 8;
                }
            } else
            if(k != selectedItem)
                return;
            break;

        case 2: // '\002'
        case 4: // '\004'
            if(k >= (display != 2 ? 8 : 7))
                return;
            if(flag)
            {
                selectedItem = k;
                selectRect.x = leftOffset + k * itemWidth;
                selectRect.y = 4;
                selectRect.width = itemWidth;
                selectRect.height = height - 8;
                break;
            }
            if(k != selectedItem)
                return;
            break;

        case 3: // '\003'
            if(k > 1)
                return;
            if(flag)
            {
                selectedItem = k;
                selectRect.x = leftOffset + k * itemWidth;
                selectRect.y = 4;
                selectRect.width = itemWidth;
                selectRect.height = height - 8;
                break;
            }
            if(k != selectedItem)
                return;
            break;
        }
        inSelection = true;
    }

    public boolean mouseDown(Event event, int i, int j)
    {
        checkPoint(i, j, true);
        if(inSelection)
            repaint(selectRect.x, selectRect.y, selectRect.width, selectRect.height);
        return true;
    }

    public boolean mouseDrag(Event event, int i, int j)
    {
        if(selectedItem < 0)
            return true;
        boolean flag = inSelection;
        checkPoint(i, j, false);
        if(flag != inSelection)
            repaint(selectRect.x, selectRect.y, selectRect.width, selectRect.height);
        return true;
    }

    public boolean mouseUp(Event event, int i, int j)
    {
        if(selectedItem < 0)
            return true;
        checkPoint(i, j, false);
        if(inSelection)
            if(display == 5)
                owner.doPaletteClick(selectedItem - 1);
            else
                owner.doPaletteClick(selectedItem);
        inSelection = false;
        selectedItem = -1;
        repaint(selectRect.x, selectRect.y, selectRect.width, selectRect.height);
        return true;
    }

    static final Color backgroundColor = new Color(220, 220, 255);
    static final Color textColor;
    static final Color itemColor;
    static final Color hitColor;
    static final Color hiliteColor = new Color(150, 255, 255);
    static final int NONE = 0;
    static final int STATES = 1;
    static final int SYMBOLS = 2;
    static final int DIRECTIONS = 3;
    static final int SYMBOLSANDDEFAULT = 4;
    static final int STATESANDHALT = 5;
    int display;
    int width;
    int height;
    int leftOffset;
    int baseLine;
    int itemWidth;
    boolean twoRowsOfStates;
    int selectedItem;
    Rectangle selectRect;
    boolean inSelection;
    MachinePanel owner;
    Font font;
    FontMetrics fm;
    Font smallFont;
    FontMetrics smallFm;

    static 
    {
        textColor = Color.blue;
        itemColor = Color.red;
        hitColor = Color.pink;
    }
}
