// Decompiled by Jad v1.5.7d. Copyright 2000 Pavel Kouznetsov.
// Jad home page: http://www.geocities.com/SiliconValley/Bridge/8617/jad.html
// Decompiler options: packimports(3) 
// Source File Name:   MachineData.java

package tm;

import java.io.*;

// Referenced classes of package tmcm.xTuringMachine:
//            MachineInputException, Rule

class MachineData
{

    protected class Rule {
       // Rule() {
       //     return;
       // }

        int state;
        char symbol;
        int newState;
        char newSymbol;
        boolean direction;
    }

    MachineData()
    {
        ruleList = new Rule[200];
        ruleCt = 0;
        newSymbol = new char[25][8];
        moveDirection = new boolean[25][8];
        newState = new int[25][8];
        tape_pos = new StringBuffer();
        tape_neg = new StringBuffer();
        clearRules();
        tape_neg.append(' ');
        for(int i = 0; i < 200; i++)
            ruleList[i] = new Rule();

    }

    void clearRules()
    {
        for(int i = 0; i < 25; i++)
        {
            for(int j = 0; j < 8; j++)
            {
                newState[i][j] = 999;
                newSymbol[i][j] = '_';
                moveDirection[i][j] = false;
            }

        }

        ruleCt = 0;
    }

    void clearTape()
    {
        tape_pos.setLength(0);
        tape_neg.setLength(1);
    }

    int getNewState(int i, char c)
    {
        int j = "_$01xyz*".indexOf(c);
        if(j == -1)
            return -1;
        int k = newState[i][j];
        if(k != 999)
            return k;
        else
            return newState[i][7];
    }

    char getNewSymbol(int i, char c)
    {
        int j = "_$01xyz*".indexOf(c);
        if(j == -1)
            return '?';
        if(newState[i][j] == 999)
        {
            char c1 = newSymbol[i][7];
            if(c1 == '*')
                return c;
            else
                return newSymbol[i][7];
        } else
        {
            return newSymbol[i][j];
        }
    }

    boolean getDirection(int i, char c)
    {
        int j = "_$01xyz*".indexOf(c);
        if(j == -1)
            return true;
        if(newState[i][j] == 999)
            return moveDirection[i][7];
        else
            return moveDirection[i][j];
    }

    void setActionData(int i, char c, char c1, boolean flag, int j)
    {
        int k;
        if(c == '*')
        {
            k = 7;
        } else
        {
            k = "_$01xyz*".indexOf(c);
            if(k == -1)
                return;
        }
        setRuleListData(i, c, c1, flag, j);
        newState[i][k] = j;
        newSymbol[i][k] = c1;
        moveDirection[i][k] = flag;
    }

    void deleteRule(int i, char c)
    {
        int j;
        if(c == '*')
        {
            j = 7;
        } else
        {
            j = "_$01xyz*".indexOf(c);
            if(j == -1)
                return;
        }
        removeFromRuleList(i, c);
        newState[i][j] = 999;
    }

    void setTape(int i, char c)
    {
        if(i >= 0)
        {
            if(i < tape_pos.length())
                tape_pos.setCharAt(i, c);
            else
            if(c != '_')
            {
                for(int j = tape_pos.length(); j < i; j++)
                    tape_pos.append('_');

                tape_pos.append(c);
            }
        } else
        {
            i = -i;
            if(i < tape_neg.length())
                tape_neg.setCharAt(i, c);
            else
            if(c != '_')
            {
                for(int k = tape_neg.length(); k < i; k++)
                    tape_neg.append('_');

                tape_neg.append(c);
            }
        }
    }

    char getTape(int i)
    {
        if(i >= 0)
            if(i >= tape_pos.length())
                return '_';
            else
                return tape_pos.charAt(i);
        i = -i;
        if(i >= tape_neg.length())
            return '_';
        else
            return tape_neg.charAt(i);
    }

    int firstFilledSquare()
    {
        int i;
        for(i = tape_neg.length() - 1; i > 0 && tape_neg.charAt(i) == '_'; i--);
        if(i == 0)
        {
            for(i = 0; i < tape_pos.length() && tape_pos.charAt(i) == '_'; i++);
            if(i < tape_pos.length())
                return i;
            else
                return 0;
        } else
        {
            return -i;
        }
    }

    int lastFilledSquare()
    {
        int i;
        for(i = tape_pos.length() - 1; i >= 0 && tape_pos.charAt(i) == '_'; i--);
        if(i < 0)
        {
            for(i = 1; i < tape_neg.length() && tape_neg.charAt(i) == '_'; i++);
            if(i < tape_neg.length())
                return -i;
            else
                return 0;
        } else
        {
            return i;
        }
    }

    private void setRuleListData(int i, char c, char c1, boolean flag, int j)
    {
        int k = 0;
        if(c == '*')
            for(; k < ruleCt && (ruleList[k].state < i || ruleList[k].state == i && ruleList[k].symbol != c); k++);
        else
            for(; k < ruleCt && (ruleList[k].state < i || ruleList[k].state == i && ruleList[k].symbol != '*' && ruleList[k].symbol < c); k++);
        if(k == ruleCt)
        {
            ruleCt++;
            ruleList[k].state = i;
            ruleList[k].symbol = c;
        } else
        if(ruleList[k].state != i || ruleList[k].symbol != c)
        {
            Rule rule = ruleList[ruleCt];
            for(int l = ruleCt; l > k; l--)
                ruleList[l] = ruleList[l - 1];

            ruleCt++;
            ruleList[k] = rule;
            ruleList[k].state = i;
            ruleList[k].symbol = c;
        }
        ruleList[k].newState = j;
        ruleList[k].newSymbol = c1;
        ruleList[k].direction = flag;
    }

    private void removeFromRuleList(int i, char c)
    {
        int j;
        for(j = 0; j < ruleCt && (ruleList[j].state < i || ruleList[j].state == i && ruleList[j].symbol != c); j++);
        if(j < ruleCt && ruleList[j].state == i && ruleList[j].symbol == c)
        {
            Rule rule = ruleList[j];
            for(int k = j; k < ruleCt - 1; k++)
                ruleList[k] = ruleList[k + 1];

            ruleList[ruleCt - 1] = rule;
            ruleCt--;
        }
    }

    int getRuleCount()
    {
        return ruleCt;
    }

    Rule getRule(int i)
    {
        if(i < 0 || i >= ruleCt)
            return null;
        else
            return ruleList[i];
    }

    int findRule(int i, char c)
    {
        int j;
        for(j = 0; j < ruleCt && (ruleList[j].state < i || ruleList[j].state == i && ruleList[j].symbol != c); j++);
        if(j < ruleCt && ruleList[j].state == i && ruleList[j].symbol == c)
            return j;
        else
            return -1;
    }

    boolean ruleDefined(int i, char c)
    {
        int j = "_$01xyz*".indexOf(c);
        if(j < 0)
            return false;
        else
            return newState[i][j] != 999;
    }

    void write(PrintStream printstream, int i)
    {
        printstream.println("xTuringMachine File Format 1.0");
        if(printstream.checkError())
            return;
        printstream.println("_$01xyz* 25");
        int j = firstFilledSquare();
        int k = lastFilledSquare();
        printstream.println("" + j + ' ' + k + ' ' + i + ';');
        int l = 0;
        for(int i1 = j; i1 <= k; i1++)
        {
            if(l == 50)
            {
                printstream.println();
                l = 0;
            }
            printstream.print(getTape(i1));
            l++;
        }

        printstream.println();
        printstream.println("" + ruleCt + ';');
        for(int j1 = 0; j1 < ruleCt; j1++)
            printstream.println("" + ruleList[j1].state + ' ' + ruleList[j1].symbol + ' ' + ruleList[j1].newSymbol + ' ' + (ruleList[j1].direction ? 'R' : 76) + ' ' + ruleList[j1].newState + ';');

    }

    void read(InputStream inputstream)
        throws MachineInputException
    {
        clearRules();
        try
        {
            DataInputStream datainputstream = new DataInputStream(inputstream);
            String s = datainputstream.readLine();
            if(!s.trim().equalsIgnoreCase("xTuringMachine File Format 1.0"))
                throw new MachineInputException("Not a legal input file (missing header on line 1)");
            s = datainputstream.readLine();
            if(!s.trim().equalsIgnoreCase("_$01xyz* 25"))
                throw new MachineInputException("Not a legal input file (illegal list of symbols or number of states in line 2)");
            int i = getInt(datainputstream, 3);
            int j = getInt(datainputstream, 3);
            if(i > j)
                throw new MachineInputException("Illegal data.  (First tape square comes after last tape square.)");
            setTape(i, '_');
            setTape(j, '_');
            saveCurrentSquare = getInt(datainputstream, 3);
            datainputstream.readLine();
            for(int k = i; k <= j; k++)
            {
                int l;
                do
                    l = datainputstream.read();
                while(l == 13 || l == 10);
                if(l == -1)
                    throw new MachineInputException("Illegal input.  (Number of tape symbols provided is less than number specified.)");
                int j1 = "_$01xyz*".indexOf(l);
                if(j1 < 0 || l == 42)
                    throw new MachineInputException("Illegal input.  Illegal tape symbol specified: " + (char)l + '.');
                setTape(k, (char)l);
            }

            datainputstream.readLine();
            int i1 = getInt(datainputstream, 4);
            if(i1 < 0)
                throw new MachineInputException("Illegal input.  The number of rules specified is less than zero.");
            if(i1 > 200)
                throw new MachineInputException("Illegal input.  The number of rules specified is larger than the maximum.");
            datainputstream.readLine();
            for(int k1 = 0; k1 < i1; k1++)
            {
                int l1 = getState(datainputstream, k1 + 5);
                if(l1 == -1)
                    throw new MachineInputException("Illegal input.  Illegal rule found on line " + (k1 + 5) + '.');
                char c = getSymbol(datainputstream, k1 + 5);
                char c1 = getSymbol(datainputstream, k1 + 5);
                if(c1 == '*' && c != '*')
                    throw new MachineInputException("Illegal input.  Illegal rule found on line " + (k1 + 5) + '.');
                boolean flag = getDirection(datainputstream, k1 + 5);
                int i2 = getState(datainputstream, k1 + 5);
                datainputstream.readLine();
                setActionData(l1, c, c1, flag, i2);
            }

        }
        catch(IOException ioexception)
        {
            throw new MachineInputException("Input error occured while reading from file. (" + ioexception + ")");
        }
    }

    int getInt(DataInputStream datainputstream, int i)
        throws MachineInputException, IOException
    {
        boolean flag = false;
        int j;
        do
            j = datainputstream.read();
        while(j == 32 || j == 9);
        if(j == 45)
        {
            flag = true;
            j = datainputstream.read();
        }
        if(j == -1)
            throw new MachineInputException("Unexpected end of file encountered while reading rules from file.");
        if(j > 57 || j < 48)
            throw new MachineInputException("Illegal data found while looking for integer on line " + i + ".");
        int k = 0;
        do
        {
            k = (10 * k + j) - 48;
            j = datainputstream.read();
        } while(j >= 48 && j <= 57);
        if(flag)
            return -k;
        else
            return k;
    }

    int getState(DataInputStream datainputstream, int i)
        throws MachineInputException, IOException
    {
        boolean flag = false;
        int j;
        do
            j = datainputstream.read();
        while(j == 32 || j == 9);
        if(j == -1)
            throw new MachineInputException("Unexpected end of file encountered while reading rules from file.");
        if(j == 45)
        {
            flag = true;
            j = datainputstream.read();
        }
        if(j > 57 || j < 48)
            throw new MachineInputException("Illegal state specification found while reading rule on line  " + i + ".");
        int k = 0;
        do
        {
            k = (10 * k + j) - 48;
            j = datainputstream.read();
        } while(j >= 48 && j <= 57);
        if(flag)
            k = -k;
        if(k == -1)
            return -1;
        if(k >= 0 && k < 25)
            return k;
        else
            throw new MachineInputException("Illegal state specification found while reading rule on line  " + i + ".");
    }

    char getSymbol(DataInputStream datainputstream, int i)
        throws MachineInputException, IOException
    {
        int j;
        do
            j = datainputstream.read();
        while(j == 32 || j == 9);
        if(j == -1)
            throw new MachineInputException("Unexpected end of file encountered while reading rules from file.");
        if("_$01xyz*".indexOf(j) >= 0)
            return (char)j;
        else
            throw new MachineInputException("Illegal symbol found while reading rule on line " + i + ".");
    }

    boolean getDirection(DataInputStream datainputstream, int i)
        throws MachineInputException, IOException
    {
        int j;
        do
            j = datainputstream.read();
        while(j == 32 || j == 9);
        if(j == -1)
            throw new MachineInputException("Unexpected end of file encountered while reading rules from file.");
        if(j == 76 || j == 108)
            return false;
        if(j == 82 || j == 114)
            return true;
        else
            throw new MachineInputException("Illegal direction specification found while reading rule on line " + i + ".");
    }

    static final int STATES = 25;
    static final int SYMBOLS = 8;
    static final String symbolNames = "_$01xyz*";
    static final int UNSPECIFIED = 999;
    static final int HALTSTATE = -1;
    static final int DEFAULT = 7;
    private Rule ruleList[];
    private int ruleCt;
    private char newSymbol[][];
    private boolean moveDirection[][];
    private int newState[][];
    private StringBuffer tape_pos;
    private StringBuffer tape_neg;
    int saveCurrentSquare;
}
