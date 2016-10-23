

import java.io.Serializable;

import jp.tradesc.superkaburobo.sdk.trade.data.Stock;

/**
 * 銘柄単位のメモ情報を格納するクラスです。<br>
 * 以下の情報を取り扱います。<br>
 * <ul>
 * <li>銘柄</li>
 * <li>買い・売り区分</li>
 * <li>売買理由</li>
 * </ul>
 * @author (c) 2004-2008 kaburobo.jp and Trade Science Corp. All rights reserved.
 */
public class SampleObjectRecord implements Serializable{
	private static final long serialVersionUID = 6825887897748591904L;
	/** 対象銘柄です, 基本的に null はありませんが、未初期化の場合は null になります。 */
	private Stock   stock = null;
	/** 買いシグナルか売りシグナルかを示します。<br>true: 買い<br>false: 失敗 */
	private boolean isBuy = true;
	/** シグナルの強さを絶対値で表します。 */
	private double  signalStrengh = 0;
	/** シグナル発生理由です。 */
	private String  reason = "";
	
	/**
	 * コンストラクタです。
	 * @param stock  対象銘柄
	 * @param isBuy  買いか売りか、trueで買い、falseで売り
	 * @param signalStrength シグナルの強さ(絶対値)
	 * @param reason シグナル発生理由
	 */
	public SampleObjectRecord(Stock stock, boolean isBuy, double signalStrength, String reason){
		this.stock         = stock;
		this.isBuy         = isBuy;
		this.signalStrengh = signalStrength;
		this.reason        = reason;
	}
	
	/**
	 *  対象銘柄を取得します。
	 * @return 対象銘柄
	 */
	public Stock   getStock(){
		return stock;
	}
	
	/** 買いか売りかを取得します。
	 * @return 買いか売りか<br>
	 * true: 買い、false: 売り
	 */
	public boolean isBuy(){
		return isBuy;
	}
	
	/**
	 * シグナルの強さを取得します。
	 * @return シグナルの強さ。<br>
	 * サンプルコードでは売りと買いの区別はありません。<br>
	 */
	public double getStrength(){
		return signalStrengh;
	}
	
	/**
	 * シグナル発生理由を取得します。
	 * @return シグナル発生理由
	 */
	public String getReason(){
		return reason;
	}
}
