// Decompiled by Jad v1.5.7d. Copyright 2000 Pavel Kouznetsov.
// Jad home page: http://www.geocities.com/SiliconValley/Bridge/8617/jad.html
// Decompiler options: packimports(3) 
// Source File Name:   xTuringMachineApplet.java

//package tm;

import java.applet.Applet;
import java.awt.*;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.Vector;
//import TuringMachinePanel;

public class TuringMachineApplet extends Applet {
    
    public String getAppletInfo() {
        return "xTuringMachine, by David J. Eck (eck@hws.edu), Version 1.0 August 1997.";
    }
    
    public String[][] getParameterInfo() {
        return parameterInfo;
    }
    
    public void init() {
        setBackground(Color.lightGray);
        setLayout(new BorderLayout());
        Object aobj[] = getURLs();
        if(aobj == null)
            TMpanel = new TuringMachinePanel(null, null);
        else
            TMpanel = new TuringMachinePanel((URL[])aobj[0], (String[])aobj[1]);
        add("Center", TMpanel);
    }

    public void start() {
        TMpanel.start();
    }

    public void stop() {
        TMpanel.stop();
    }

    public void destroy() {
        TMpanel.destroy();
    }

    Object[] getURLs() {
        int i = 0;
        Vector vector = new Vector();
        Vector vector1 = new Vector();
        String s = getParameter("BASE");
        URL url;
        if(s == null)
            url = getDocumentBase();
        else
            try
            {
                url = new URL(getDocumentBase(), s);
            }
            catch(MalformedURLException _ex)
            {
                return null;
            }
        String s1 = getParameter("URL");
        do
        {
            if(s1 != null)
            {
                URL url1;
                try
                {
                    url1 = new URL(url, s1);
                }
                catch(MalformedURLException _ex)
                {
                    continue;
                }
                vector.addElement(url1);
                vector1.addElement(s1);
            }
            i++;
            s1 = getParameter("URL" + i);
        } while(s1 != null);
        if(vector.size() > 0)
        {
            URL aurl[] = new URL[vector.size()];
            String as[] = new String[vector.size()];
            for(int j = 0; j < aurl.length; j++)
            {
                aurl[j] = (URL)vector.elementAt(j);
                as[j] = (String)vector1.elementAt(j);
            }

            Object aobj[] = new Object[2];
            aobj[0] = aurl;
            aobj[1] = as;
            return aobj;
        } else
        {
            return null;
        }
    }

   // public TuringMachineApplet()
   // {
   // }

    TuringMachinePanel TMpanel;
    String parameterInfo[][] = {
        {
            "URL", "url", "absolute or relative url of a text file containing a sample xTuringMachine program"
        }, {
            "URL1,URL2,...", "url", "additional URLs of xTTuringMachine programs"
        }, {
            "BASE", "url", "base url for interpreting URL, URL1, ...; if not given, document base is used"
        }
    };
}
