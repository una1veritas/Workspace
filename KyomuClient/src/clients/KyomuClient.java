package clients;
//
//  KyomuClient.java
//  KyomuClient
//
//  Created by Sin Shimozono on 11/08/19.
//  Copyright (c) 2011, ???????????. All rights reserved.
//
//  A simple Web Start application
//

//import java.awt.*;
import java.net.*;
//import javax.swing.*;
//import javax.jnlp.*;
//import java.util.*;
//import java.util.jar.*;
//import java.io.IOException;
//import java.lang.reflect.*;

import clients.*;

public class KyomuClient extends URLClassLoader {
	static final String[] SERVER_URI = {
			"kyomuinfo://131.206.103.7:3401/TeacherTool/",
			"kyomuinfo://131.206.103.7:3402/StudentTool/",
			"kyomuinfo://131.206.103.7:3404/KyomuTool/" };

	public static void main(String args[]) throws Exception {
		URI toolURI = null;
		String toolName = "TeacherTool";
		if (args.length == 1) {
			toolName = args[0];
		}
		for (int i = 0; i < SERVER_URI.length; i++) {
			toolURI = new URI(SERVER_URI[i]);
			if (toolURI.getPath().startsWith(toolName,1)) 
				break;
		}
		System.err.println("Selected tool server: "+toolURI);
		if (toolURI == null) {
			System.err.println("Cannot assume requested service. ");
			System.exit(1);
		}

		System.out.println("Starting this client locally...");
		KyomuTool client = new KyomuTool(toolURI);
		/*
		// openWithBrowser();
		try {
			System.out.println("Attempting to open an url in your browser.");
			URL apple = new URL(
					"http://daisyserver.daisy.ai.kyutech.ac.jp/~sin/kyomu/");
			BasicService bs = (BasicService) ServiceManager
					.lookup("javax.jnlp.BasicService");
			bs.showDocument(apple);
		} catch (UnavailableServiceException unavailableserviceexception) {
			JOptionPane
					.showMessageDialog(
							null,
							"This functionality is only available when this application is launched with Java Web Start.",
							"", 0);
		} catch (MalformedURLException malformedurlexception) {
		}
		*/
	}

	public KyomuClient(URL url) {
		super(new URL[] { url });
	}

	public KyomuClient(String urladdr) throws Exception {
		super(new URL[] { new URL(urladdr) });
	}
/*
	public String getMainClassName(URL url) throws IOException {
		URL u = new URL("jar", "", url + "!/");
		JarURLConnection uc = (JarURLConnection) u.openConnection();
		Attributes attr = uc.getMainAttributes();
		return attr != null ? attr.getValue(Attributes.Name.MAIN_CLASS) : null;
	}
	*/
/*
	public void invokeRemoteClass(String name, String[] args)
			throws ClassNotFoundException, NoSuchMethodException,
			InvocationTargetException, IllegalAccessException {
		/*
		 * System.out.println(); System.out.println("Class "+ name +
		 * " methods: "); Method[] methods = c.getMethods(); for (int i = 0; i <
		 * methods.length; i++) { Method method = methods[i];
		 * System.out.println(method.getName()); }
		 */
		//
	/*
		Method m = loadClass(name).getMethod("main",
				new Class[] { args.getClass() });
		m.setAccessible(true);
		// int mods = m.getModifiers();
		// if (m.getReturnType() != void.class || !Modifier.isStatic(mods) ||
		// !Modifier.isPublic(mods)) {
		// throw new NoSuchMethodException("main");
		// }
		m.invoke(null, new Object[] { args });
	}
*/

}
