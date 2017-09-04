// Decompiled by Jad v1.5.7d. Copyright 2000 Pavel Kouznetsov.
// Jad home page: http://www.geocities.com/SiliconValley/Bridge/8617/jad.html
// Decompiler options: packimports(3) 
// Source File Name:   MachineData.java

//package tm;

import java.io.*;
import java.net.URL;
import java.net.URLConnection;

// Referenced classes of package tmcm.xTuringMachine:
//            MachineData, MachineInputException, MachinePanel

class MachineLoader extends Thread
{

    MachineLoader(URL url1, MachinePanel machinepanel, int i)
    {
        url = url1;
        owner = machinepanel;
        ID = i;
        start();
    }

    MachineLoader(String s, String s1, MachinePanel machinepanel, int i)
    {
        fileName = s1;
        directory = s;
        owner = machinepanel;
        ID = i;
        start();
    }

    public void run()
    {
        try
        {
            setPriority(getPriority() - 1);
        }
        catch(RuntimeException _ex) { }
        try
        {
            Thread.sleep(500 + (int)(500D * Math.random()));
        }
        catch(InterruptedException _ex) { }
        Object obj = null;
        try
        {
            if(url != null)
                obj = url.openConnection().getInputStream();
            else
                obj = new FileInputStream(new File(directory, fileName));
            MachineData machinedata = new MachineData();
            machinedata.read(((InputStream) (obj)));
            owner.doneLoading(machinedata, ID);
        }
        catch(MachineInputException machineinputexception)
        {
            errorMessage = "LOAD FAILED:  " + machineinputexception.getMessage();
            owner.loadingError(ID);
        }
        catch(SecurityException securityexception)
        {
            errorMessage = "LOAD FAILED, SECURITY ERROR:  " + securityexception.getMessage();
            owner.loadingError(ID);
        }
        catch(Exception exception1)
        {
            errorMessage = "LOAD FAILED, ERROR:  " + exception1.toString();
            owner.loadingError(ID);
        }
        finally
        {
            if(obj != null)
                try
                {
                    ((InputStream) (obj)).close();
                }
                catch(IOException _ex) { }
        }
    }

    MachinePanel owner;
    int ID;
    URL url;
    String fileName;
    String directory;
    String errorMessage;
}
