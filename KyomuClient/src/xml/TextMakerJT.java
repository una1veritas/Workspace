package xml;

public class TextMakerJT implements IJTVisitor {
	private StringBuffer item;
	private StringBuffer itemContent;
	private StringBuffer desc;
	private boolean itemizeFlag = false;
	private boolean enumerateFlag = false;
	private int enumerateCount = 0;

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
		}

		if (tag.equals("講義内容") || tag.equals("位置付け") || tag.equals("講義項目")
				|| tag.equals("進め方") || tag.equals("評価方法") || tag.equals("備考")
				|| tag.equals("教科書") || tag.equals("参考書")
				|| tag.equals("キーワード") 
				|| tag.equals("授業の概要")
				|| tag.equals("カリキュラムにおけるこの授業の位置付け") || tag.equals("授業項目")
				|| tag.equals("授業の進め方") || tag.equals("授業の達成目標")
				|| tag.equals("成績評価の基準および評価方法")) {

			if (tag.equals("授業項目")) {
				tag = "授業項目 (授業計画)";
			}
			if (tag.equals("授業の達成目標")) {
				tag = "授業の達成目標 (学習・教育目標との関連)";
			}

			item = new StringBuffer();
			itemContent = new StringBuffer();
			item.append("\n" + tag + " \n");

		} else if (tag.equals("UL")) {

			itemizeFlag = true;

		} else if (tag.equals("OL")) {

			enumerateFlag = true;
			enumerateCount = 0;

		} else if (tag.equals("LI")) {

			if (itemizeFlag) {
				itemContent.append("  ・ " + node.getValue() + "\n");
			} else if (enumerateFlag) {
				enumerateCount++;
				itemContent.append("  (" + enumerateCount + ") "
						+ node.getValue() + "\n");
			}

		} else if (tag.equals("P")) {

			itemContent.append("  " + node.getValue() + "\n");

		}
	}

	public void leave(JTElement node) {
		String tag = node.getName();

		if (tag.equals("講義内容") || tag.equals("位置付け") || tag.equals("講義項目")
				|| tag.equals("進め方") || tag.equals("評価方法") || tag.equals("備考")
				|| tag.equals("教科書") || tag.equals("参考書")
				|| tag.equals("キーワード") || tag.equals("授業の概要")
				|| tag.equals("カリキュラムにおけるこの授業の位置付け") || tag.equals("授業項目")
				|| tag.equals("授業の進め方") || tag.equals("授業の達成目標")
				|| tag.equals("成績評価の基準および評価方法")) {

			String content = itemContent.toString();
			if (content.length() > 0) {
				desc.append(item.toString()).append(content);
			} else {
				desc.append(item.toString()).append("\n");
			}
		} else if (tag.equals("OL")) {

			enumerateFlag = false;

		} else if (tag.equals("UL")) {

			itemizeFlag = false;
		}
	}

	public void enter(JTNode node) {
		throw (new InternalError());
	}

	public void leave(JTNode node) {
		throw (new InternalError());
	}
}
