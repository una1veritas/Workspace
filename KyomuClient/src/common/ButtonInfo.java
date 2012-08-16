package common;

public class ButtonInfo {
  //**** commandType : SIMPLE, DELETE, UPDATE, INSERT, SPECIAL ****//
  public String commandType;

  //**** if commandType == SIMPLE ****//
  public String methodToInvoke;
  public String methodParam;

  //**** if commandType == DELETE, UPDATE, INSERT, SPECIAL ****//
  public String serviceName;
  public String commandCode;
  
  public ButtonInfo(String commandType,
		    String param1,
		    String param2) {    
    this.commandType = commandType;
    if (commandType.equals("SIMPLE")) {
      methodToInvoke = param1;
      methodParam = param2;
    } else {
      serviceName = param1;
      commandCode = param2;
    }
  }
}
