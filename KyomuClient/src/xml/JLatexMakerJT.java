package xml;

public class JLatexMakerJT implements IJTVisitor {
  private StringBuffer item; 
  private StringBuffer itemContent;
  private StringBuffer desc;
 
  public String getText() {
    return (new String(desc));
  }

  public void enter(JTAttr node) {
    // do nothing
  }
  
  public void leave(JTAttr node) {
    // do nothing
  }
  
  public void enter(JTElement node) {
    String tag = node.getName();

    if (tag.equals("講義")) {
      desc = new StringBuffer();
      desc.append("\\begin{description}\n");
    }
    
    if (tag.equals("講義内容") || tag.equals("位置付け") ||
        tag.equals("講義項目") || tag.equals("進め方") ||
        tag.equals("評価方法") || tag.equals("備考") ||
        tag.equals("教科書") || tag.equals("参考書") || 
	tag.equals("キーワード") ||
	tag.equals("授業の概要") ||
	tag.equals("カリキュラムにおけるこの授業の位置付け") ||
	tag.equals("授業項目") ||
	tag.equals("授業の進め方") ||
	tag.equals("授業の達成目標") ||
	tag.equals("成績評価の基準および評価方法")) {    

      if (tag.equals("授業項目")) {
	tag = "授業項目 (授業計画)";
      }
      if (tag.equals("授業の達成目標")) {
	tag = "授業の達成目標 (学習・教育目標との関連)";
      }
              
      item = new StringBuffer();
      itemContent = new StringBuffer();        
      item.append("\\item [" + tag + "] \\ \n\n");      
    } else if (tag.equals("UL")) {
      itemContent.append("\\begin{itemize}\n");
    } else if (tag.equals("OL")) {
      itemContent.append("\\begin{enumerate} \\renewcommand{\\labelenumi}{(\\arabic{enumi}) } \\vspace{-0.1cm} \n");
    } else if (tag.equals("LI")) {  
      String text2;
      String text = node.getValue();
      if (text.indexOf("%") >= 0) {
	text2 = text.replace("%", "\\%");
	text = text2;
      }
      if (text.indexOf("_") >= 0) {
	text2 = text.replace("_", "\\_");
	text = text2;
      }
      if (text.indexOf("&") >= 0) {
	text2 = text.replace("&", "＆");
	text = text2;
      }  
      if (text.indexOf("~") >= 0) {
	text2 = text.replace("~", "\\~");
	text = text2;
      }   
      if (text.indexOf("<") >= 0) {
	text2 = text.replace("<", " [");
	text = text2;
      }    
      if (text.indexOf(">") >= 0) {
	text2 = text.replace(">", "] ");
	text = text2;
      }  
      itemContent.append("\\item " + text + "\n");
    } else if (tag.equals("P")) {
      String text2;
      String text = node.getValue();
      if (text.indexOf("%") >= 0) {
	text2 = text.replace("%", "\\%");
	text = text2;
      }
      if (text.indexOf("_") >= 0) {
	text2 = text.replace("_", "\\_");
	text = text2;
      }
      if (text.indexOf("&") >= 0) {
	text2 = text.replace("&", "\\&");
	text = text2;
      }  
      if (text.indexOf("~") >= 0) {
	text2 = text.replace("~", "\\~");
	text = text2;
      }   
      if (text.indexOf("<") >= 0) {
	text2 = text.replace("<", " [");
	text = text2;
      }    
      if (text.indexOf(">") >= 0) {
	text2 = text.replace(">", "] ");
	text = text2;
      }   
      itemContent.append("" + text + "\n\n");
    }
  }
  
  public void leave(JTElement node) {
    String tag = node.getName();

    if (tag.equals("講義")) {
      if (!desc.toString().equals("\\begin{description}\n")) {
	desc.append("\\end{description} ");
      } else {
	desc = new StringBuffer();
      }
    }
    
    if (tag.equals("講義内容") || tag.equals("位置付け") ||
        tag.equals("講義項目") || tag.equals("進め方") ||
        tag.equals("評価方法") || tag.equals("備考") ||
        tag.equals("教科書") || tag.equals("参考書") || 
	tag.equals("キーワード") ||
	tag.equals("授業の概要") ||
	tag.equals("カリキュラムにおけるこの授業の位置付け") ||
	tag.equals("授業項目") ||
	tag.equals("授業の進め方") ||
	tag.equals("授業の達成目標") ||
	tag.equals("成績評価の基準および評価方法")) {    

      String content = itemContent.toString();
      if (content.length() > 0) {
	desc.append(item.toString()).append(content);
      } else {
	desc.append(item.toString()).append("\n");
      }
    } else if (tag.equals("OL")) {
      itemContent.append("\\end{enumerate}\n");
    } else if (tag.equals("UL")) {
      itemContent.append("\\end{itemize}\n");
    }
  }
  
  public void enter(JTNode node) {
    throw (new InternalError());
  }
  
  public void leave(JTNode node) {
    throw (new InternalError());
  }
}
