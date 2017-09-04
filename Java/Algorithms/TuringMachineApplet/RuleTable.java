// Decompiled by Jad v1.5.7d. Copyright 2000 Pavel Kouznetsov.
// Jad home page: http://www.geocities.com/SiliconValley/Bridge/8617/jad.html
// Decompiler options: packimports(3) 
// Source File Name:   RuleTable.java

//package tm;

import java.awt.*;

// Referenced classes of package tmcm.xTuringMachine:
//            MachineData, MachinePanel, Rule, RuleCanvas, 
//            RuleMakerCanvas

class RuleTable extends Panel
{

    RuleTable(MachinePanel machinepanel)
    {
        setLayout(new BorderLayout());
        scroll = new Scrollbar(1);
        add("East", scroll);
        canvas = new RuleCanvas(this);
        add("Center", canvas);
        owner = machinepanel;
    }

    void setMachineData(MachineData machinedata)
    {
        data = machinedata;
        int i = machinedata != null ? machinedata.getRuleCount() : 0;
        canvas.topRule = 0;
        if(machinedata == null || canvas.visibleRules >= i)
        {
            scroll.disable();
            scroll.setValues(0, 1, 0, 1);
        } else
        {
            scroll.setValues(0, canvas.visibleRules, 0, i);
            scroll.enable();
        }
        canvas.selectedRule = -1;
        owner.deleteRuleButton.disable();
        canvas.repaint(2, canvas.lineHeight + 4, canvas.width - 4, canvas.height - canvas.lineHeight - 6);
    }

    void doDeleteRule()
    {
        if(canvas.selectedRule < 0 || data == null)
            return;
        MachineData.Rule rule = data.getRule(canvas.selectedRule);
        data.deleteRule(rule.state, rule.symbol);
        if(rule.state == owner.ruleMaker.state && rule.symbol == owner.ruleMaker.symbol)
            owner.makeButton.setLabel("Make Rule");
        canvas.selectedRule = -1;
        canvas.resetScroll();
        owner.deleteRuleButton.disable();
        owner.dropFocus(1);
    }

    void ruleAdded(int i, char c)
    {
        int j = data.findRule(i, c);
        canvas.selectedRule = j;
        canvas.selectedColumn = 0;
        owner.deleteRuleButton.enable();
        canvas.resetScroll();
    }

    void ruleChanged(int i, char c)
    {
        int j = data.findRule(i, c);
        if(j == canvas.selectedRule)
            canvas.selectedRule = -1;
        canvas.select(j, 0);
    }

    public boolean handleEvent(Event event)
    {
        if(event.id == 602 || event.id == 601 || event.id == 604 || event.id == 603 || event.id == 605)
        {
            owner.dropFocus(1);
            canvas.setStart();
            return true;
        } else
        {
            return super.handleEvent(event);
        }
    }

    Scrollbar scroll;
    RuleCanvas canvas;
    MachineData data;
    MachinePanel owner;
}
