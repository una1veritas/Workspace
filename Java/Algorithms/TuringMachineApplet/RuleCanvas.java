// Decompiled by Jad v1.5.7d. Copyright 2000 Pavel Kouznetsov.
// Jad home page: http://www.geocities.com/SiliconValley/Bridge/8617/jad.html
// Decompiler options: packimports(3) 
// Source File Name:   RuleTable.java

//package tm;

import java.awt.*;

// Referenced classes of package tmcm.xTuringMachine:
//            MachineData, MachinePanel, Palette, Rule, 
//            RuleTable

class RuleCanvas extends Canvas
{

    RuleCanvas(RuleTable ruletable)
    {
        lineHeight = -1;
        columnLoc = new int[11];
        selectedRule = -1;
        setBackground(Color.white);
        owner = ruletable;
    }

    public void paint(Graphics g)
    {
        if(lineHeight == -1 || width != size().width || height != size().height)
            setUp();
        g.setFont(font);
        g.drawRect(0, 0, width + 1, height);
        g.drawRect(1, 1, width - 1, height - 2);
        g.drawLine(0, height - 2, width, height - 2);
        g.drawLine(width - 1, 0, width - 1, height);
        g.drawLine(0, lineHeight + 2, width, lineHeight + 2);
        g.drawLine(0, lineHeight + 3, width, lineHeight + 3);
        g.drawLine(columnLoc[2], 0, columnLoc[2], height);
        g.drawLine(columnLoc[4], 0, columnLoc[4], height);
        g.drawLine(columnLoc[4] - 1, 0, columnLoc[4] - 1, height);
        g.drawLine(columnLoc[6], 0, columnLoc[6], height);
        g.drawLine(columnLoc[8], 0, columnLoc[8], height);
        int i = 2 + ((lineHeight + fm.getAscent()) - fm.getDescent()) / 2;
        g.setColor(Color.blue);
        g.drawString("In State", columnLoc[1] - fm.stringWidth("In State") / 2, i);
        g.drawString("Reading", columnLoc[3] - fm.stringWidth("Reading") / 2, i);
        g.drawString("Write", columnLoc[5] - fm.stringWidth("Write") / 2, i);
        g.drawString("Move", columnLoc[7] - fm.stringWidth("Move") / 2, i);
        g.drawString("New State", columnLoc[9] - fm.stringWidth("New State") / 2, i);
        g.setColor(Color.black);
        if(owner.data == null)
            return;
        i += lineHeight + 3;
        for(int j = topRule; i < height - 4; j++)
        {
            MachineData.Rule rule = owner.data.getRule(j);
            if(rule == null)
                return;
            if(j == selectedRule)
                g.setColor(Color.red);
            else
                g.setColor(Color.black);
            String s = String.valueOf(rule.state);
            g.drawString(s, columnLoc[1] - fm.stringWidth(s) / 2, i);
            if(rule.symbol == '*')
                s = "other";
            else
                s = String.valueOf(rule.symbol);
            g.drawString(s, columnLoc[3] - fm.stringWidth(s) / 2, i);
            if(rule.newSymbol == '*')
                s = "same";
            else
                s = String.valueOf(rule.newSymbol);
            g.drawString(s, columnLoc[5] - fm.stringWidth(s) / 2, i);
            if(rule.direction)
                s = "R";
            else
                s = "L";
            g.drawString(s, columnLoc[7] - fm.stringWidth(s) / 2, i);
            if(rule.newState == -1)
                s = "h";
            else
                s = String.valueOf(rule.newState);
            g.drawString(s, columnLoc[9] - fm.stringWidth(s) / 2, i);
            if(hasFocus && j == selectedRule)
                putHilite(g, j - topRule, selectedColumn);
            i += lineHeight;
        }

    }

    void putHilite(Graphics g, int i, int j)
    {
        g.setColor(Palette.hiliteColor);
        int k = columnLoc[4 + 2 * j] + 4;
        int l = columnLoc[6 + 2 * j] - 3 - k;
        int i1 = (i + 1) * lineHeight + 6;
        int j1 = lineHeight;
        g.drawRect(k, i1, l, j1);
        g.drawRect(k + 1, i1 + 1, l - 2, j1 - 2);
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
        lineHeight = fm.getHeight() + 5;
        visibleRules = (height - lineHeight - 6) / lineHeight;
        if(owner.data == null || visibleRules >= owner.data.getRuleCount())
        {
            topRule = 0;
            owner.scroll.disable();
            owner.scroll.setValues(0, 1, 0, 1);
        } else
        if(topRule + visibleRules >= owner.data.getRuleCount())
        {
            topRule = owner.data.getRuleCount() - visibleRules;
            owner.scroll.setValues(topRule, visibleRules, 0, topRule + visibleRules);
            owner.scroll.enable();
        } else
        {
            owner.scroll.setValues(topRule, visibleRules, 0, owner.data.getRuleCount());
            owner.scroll.enable();
        }
    }

    void select(int i, int j)
    {
        if(selectedRule == i && selectedColumn == j && (i == -1 || i >= topRule && i < topRule + visibleRules))
            return;
        if(i == -1 && (selectedRule < topRule || selectedRule >= topRule + visibleRules))
        {
            selectedRule = -1;
            return;
        }
        firstKeyDownInSelection = true;
        int k = selectedRule;
        selectedRule = i;
        if(j >= 0)
            selectedColumn = j;
        if(i == -1)
            owner.owner.deleteRuleButton.disable();
        else
            owner.owner.deleteRuleButton.enable();
        if(i >= 0 && (i < topRule || i >= topRule + visibleRules))
        {
            int l;
            if(i < topRule)
                l = i;
            else
                l = (i - visibleRules) + 1;
            owner.scroll.setValue(l);
            setStart();
        } else
        {
            if(k != -1)
                repaint(2, ((k - topRule) + 1) * lineHeight + 5, width - 3, lineHeight + 3);
            if(selectedRule != -1 && selectedRule != k)
                repaint(2, ((selectedRule - topRule) + 1) * lineHeight + 5, width - 3, lineHeight + 3);
        }
    }

    void resetScroll()
    {
        if(owner.data == null || owner.data.getRuleCount() <= visibleRules)
        {
            topRule = 0;
            owner.scroll.disable();
            owner.scroll.setValues(0, 1, 0, 1);
        } else
        {
            if(selectedRule >= 0 && selectedRule < topRule)
                topRule = selectedRule;
            else
            if(selectedRule >= topRule + visibleRules)
                topRule = (selectedRule - visibleRules) + 1;
            int i = owner.data.getRuleCount() - visibleRules;
            if(topRule > i)
                topRule = i;
            owner.scroll.setValues(topRule, visibleRules, 0, i + visibleRules);
            owner.scroll.enable();
        }
        setStart();
    }

    void setStart()
    {
        topRule = owner.scroll.getValue();
        if(topRule > owner.data.getRuleCount() - visibleRules)
        {
            topRule = Math.max(0, owner.data.getRuleCount() - visibleRules);
            owner.scroll.setValue(topRule);
        }
        repaint(2, lineHeight + 4, width - 3, height - lineHeight - 6);
    }

    void repaintSelection()
    {
        if(selectedRule == -1)
        {
            return;
        } else
        {
            int i = selectedRule - topRule;
            int j = columnLoc[4 + 2 * selectedColumn] + 1;
            int k = columnLoc[6 + 2 * selectedColumn] - j;
            int l = (i + 1) * lineHeight + 5;
            int i1 = lineHeight + 3;
            repaint(j, l, k, i1);
            return;
        }
    }

    void focusIsOn()
    {
        hasFocus = true;
        repaintSelection();
        firstKeyDownInSelection = true;
    }

    void focusIsOff()
    {
        hasFocus = false;
        repaintSelection();
    }

    void processKey(int i)
    {
        if(owner.data == null)
            return;
        MachineData.Rule rule = owner.data.getRule(selectedRule);
        if(rule == null)
            return;
        if(i >= 32 && i <= 127)
        {
            if(i == 32)
                i = 35;
            else
                i = Character.toLowerCase((char)i);
            switch(selectedColumn)
            {
            case 0: // '\0'
                if(i == 105)
                    i = 49;
                else
                if(i == 111)
                    i = 48;
                if(rule.newSymbol != i && "_$01xyz*".indexOf((char)i) >= 0 && (i != 42 || rule.symbol == '*'))
                {
                    owner.data.setActionData(rule.state, rule.symbol, (char)i, rule.direction, rule.newState);
                    repaintSelection();
                }
                break;

            case 1: // '\001'
                if(i == 108 && rule.direction)
                {
                    owner.data.setActionData(rule.state, rule.symbol, rule.newSymbol, false, rule.newState);
                    repaintSelection();
                } else
                if(i == 114 && !rule.direction)
                {
                    owner.data.setActionData(rule.state, rule.symbol, rule.newSymbol, true, rule.newState);
                    repaintSelection();
                }
                break;

            case 2: // '\002'
                if(i == 104)
                {
                    if(rule.newState != -1)
                    {
                        owner.data.setActionData(rule.state, rule.symbol, rule.newSymbol, rule.direction, -1);
                        repaintSelection();
                    }
                } else
                if(i >= 48 && i <= 57)
                {
                    int j;
                    if(rule.newState != -1)
                        j = (rule.newState * 10 + i) - 48;
                    else
                        j = i - 48;
                    if(j >= 25 || firstKeyDownInSelection)
                        j = i - 48;
                    if(j != rule.newState)
                    {
                        owner.data.setActionData(rule.state, rule.symbol, rule.newSymbol, rule.direction, j);
                        repaintSelection();
                    }
                    firstKeyDownInSelection = false;
                }
                break;
            }
        } else
        if(i == 1006 || i == 1007)
        {
            selectedColumn = i != 1006 ? selectedColumn + 1 : selectedColumn - 1;
            if(selectedColumn < 0)
                selectedColumn = 2;
            else
            if(selectedColumn > 2)
                selectedColumn = 0;
            if(selectedColumn == 0)
            {
                if(rule.symbol == '*')
                    owner.owner.requestFocus(1, 4);
                else
                    owner.owner.requestFocus(1, 2);
            } else
            if(selectedColumn == 1)
                owner.owner.requestFocus(1, 3);
            else
                owner.owner.requestFocus(1, 5);
            if(selectedRule != -1)
                repaint(columnLoc[4] + 1, ((selectedRule - topRule) + 1) * lineHeight + 5, width - 2 - columnLoc[4], lineHeight + 3);
        } else
        if(i == 1004)
        {
            int k = selectedRule - 1;
            if(k >= 0)
                select(k, -1);
        } else
        if(i == 1005)
        {
            int l = selectedRule + 1;
            if(l < owner.data.getRuleCount())
                select(l, -1);
        } else
        if(i == 1000)
            select(0, -1);
        else
        if(i == 1001)
            select(owner.data.getRuleCount() - 1, -1);
        else
        if(i == 1002)
        {
            int i1 = selectedRule - visibleRules;
            if(i1 < 0)
                i1 = 0;
            select(i1, -1);
        } else
        if(i == 1003)
        {
            int j1 = selectedRule + visibleRules;
            if(j1 >= owner.data.getRuleCount())
                j1 = owner.data.getRuleCount() - 1;
            select(j1, -1);
        }
    }

    void processPaletteClick(int i, int j)
    {
        if(j == 3)
            processKey(i != 0 ? 82 : 76);
        else
        if(j == 2 || j == 4)
            processKey("_$01xyz*".charAt(i));
        else
        if(j == 5 && selectedColumn == 2)
        {
            if(owner.data == null)
                return;
            MachineData.Rule rule = owner.data.getRule(selectedRule);
            if(rule == null)
                return;
            int k;
            if(i == -1)
                k = -1;
            else
                k = i;
            if(k == rule.newState)
                return;
            owner.data.setActionData(rule.state, rule.symbol, rule.newSymbol, rule.direction, k);
            repaintSelection();
        }
        firstKeyDownInSelection = true;
    }

    public boolean mouseDown(Event event, int i, int j)
    {
        if(owner.data == null)
            return true;
        int k = (j - lineHeight - 5) / lineHeight;
        if(k < 0 || owner.data == null)
            return true;
        int l = topRule + k;
        if(l >= topRule + visibleRules || l >= owner.data.getRuleCount())
            return true;
        int i1 = i / (width / 5) - 2;
        if(i1 < 0)
            i1 = 0;
        else
        if(i1 > 2)
            i1 = 2;
        if(l != selectedRule)
            select(l, i1);
        else
        if(selectedColumn != i1)
        {
            selectedColumn = i1;
            repaint(2, ((l - topRule) + 1) * lineHeight + 5, width - 3, lineHeight + 3);
        }
        if(i1 == 0)
        {
            MachineData.Rule rule = owner.data.getRule(l);
            if(rule.symbol == '*')
                owner.owner.requestFocus(1, 4);
            else
                owner.owner.requestFocus(1, 2);
        } else
        if(i1 == 1)
            owner.owner.requestFocus(1, 3);
        else
        if(i1 == 2)
            owner.owner.requestFocus(1, 5);
        return true;
    }

    int lineHeight;
    int width;
    int height;
    int columnLoc[];
    RuleTable owner;
    int topRule;
    int visibleRules;
    int selectedRule;
    int selectedColumn;
    boolean hasFocus;
    Font font;
    FontMetrics fm;
    boolean firstKeyDownInSelection;
}
