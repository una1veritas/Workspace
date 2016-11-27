

import java.util.ArrayList;

import jp.tradesc.superkaburobo.sdk.robot.AbstractRobot;
import jp.tradesc.superkaburobo.sdk.trade.AnalysisManager;
import jp.tradesc.superkaburobo.sdk.trade.AssetManager;
import jp.tradesc.superkaburobo.sdk.trade.EnumAnalysisSpan;
import jp.tradesc.superkaburobo.sdk.trade.InformationManager;
import jp.tradesc.superkaburobo.sdk.trade.MemoManager;
import jp.tradesc.superkaburobo.sdk.trade.OrderManager;
import jp.tradesc.superkaburobo.sdk.trade.PortfolioManager;
import jp.tradesc.superkaburobo.sdk.trade.RobotLogManager;
import jp.tradesc.superkaburobo.sdk.trade.TradeAgent;
import jp.tradesc.superkaburobo.sdk.trade.analysis.technicalindex.MovingAverage;
import jp.tradesc.superkaburobo.sdk.trade.analysis.technicalindex.RSI;
import jp.tradesc.superkaburobo.sdk.trade.data.Portfolio;
import jp.tradesc.superkaburobo.sdk.trade.data.Stock;
import jp.tradesc.superkaburobo.sdk.trade.data.StockData;

/**
 * �T���v�����{�b�g
 * 
 * @author (c) 2004-2008 kaburobo.jp and Trade Science Corp. All rights reserved.
 * 
 * 
 */
public class SampleRobot extends AbstractRobot {
	@Override
	public void order(TradeAgent tradeAgent) {	
		// �ۑ����Ă��郁�����擾���܂��B
		SampleObjectMemo memo = (SampleObjectMemo) tradeAgent.getMemoManager().getObjectMemo();
		// ������ null �̎��̂��Ƃ��l���� null �`�F�b�N�����܂��B
		if(memo == null) return;
		
		// ���Δ���������K�v�ɉ����čs���܂��B
		orderReverseByRate(15, -6, 60); // 15%�ŗ��m���A-6%�ő��؂肵�A60���ۗL�ŋ������Δ������܂��B
		orderReverseBySignal(memo);     // �V�O�i���Ɣ��΂̃|�W�V�������������܂��B
		
		// �V�K�������s���܂��B
		orderNew(memo);
		
		// ���ɓ������̂��w�����Ȃ��悤�ɁA�����̒����\�胊�X�g���폜���܂��B
		// �O��ɖ�肵�Ȃ����������͎����I�Ɍ��Ɉ����p����܂��̂Ŗ�肠��܂���B
		memo.setMemoList(null);
		tradeAgent.getMemoManager().setObjectMemo(memo);
	}
	
	/**
	 * �V�K�����������s���܂��B
	 * @param memo �����I�u�W�F�N�g
	 */
	protected void orderNew(SampleObjectMemo memo){
		// �����Y��10�p�[�Z���g���P�����ɓ����������Ƃ��܂��B
		long orderMoney = (long)(AssetManager.getInstance().getTotalAssetValue() * 0.1);
		
		// �e��}�l�[�W�����擾���܂��B
		InformationManager im = InformationManager.getInstance();
		OrderManager       om = OrderManager.getInstance();
		
		// �X�N���[�j���O���ʂ̖����𒍕����܂��B
		for (SampleObjectRecord stockmemo: memo.getMemoList()) {
			Stock stock = stockmemo.getStock();
			
			// ���łɕۗL���Ă�������̏ꍇ�X�L�b�v���܂��B
			if(isHolding(stock)) continue;
			
			// ��������̏I�l���擾���܂��B���s�����炻�̖����̓X�L�b�v���܂��B
			StockData stockSession = im.getStockSession(stock);
			if(stockSession == null) continue;
			if(stockSession.getClosingPrice() == null) continue;
			int closingPrice = stockSession.getClosingPrice();
			
			// �I�l����w��(���p)�������T�Z�ŋ��߁A���m�Ɋۂ߂��������Z�o���܂��B
			// �󔄂莞�ł��A���ʂ͐�Βl�ɂȂ�܂��B
			int qty = om.checkRoundUnit(stock, (int)(orderMoney / closingPrice));
			
			// �ۂ߂����ʂ�0�̎��͒������܂���B
			if (qty <= 0) continue;
			
			// �����̐��ۂ��i�[����ϐ� (true:����, false:���s)
			boolean result;
			
			// ���������܂��B
			if (stockmemo.isBuy()) {
				// �������������܂��B
				
				// �������s����
				result = om.orderActualNowMarket(stock, qty);
				
			} else {
				// �󔄂�����܂��B�󔄂�̓A�b�v�e�B�b�N���[���ɂ��w�l�݂̂ł��B
				
				// �I�l����Ăђl�𒲂ׂ܂�
				int spread = om.checkSpread(closingPrice);
				
				// �w�l�̓A�b�v�e�B�b�N���[�����ᔽ���Ȃ��悤�ɁA�O���I�l�ȉ��ł͎w���܂���B
				// �O���I�l+1Tick�̉��i�Ŏw���܂��B
				result = om.orderCreditNowLimit(
						stock, 
						closingPrice + spread,
						-1 * qty); // ���蒍�����o���ꍇ�́A�}�C�i�X�̊�������͂��܂�
			}
			
			// �����̐��ۂɂ���ďo�͂�ς��܂��B
			// �Ȃ��ALastOrderResult�ɂ���Ď��s�������̗��R���킩��܂��B
			String retMsg = "";       // ����/���s
			if (result){
				retMsg = "����";
				
				// �����ɐ��������̂ŁA�����̗��R��ݒ肵�܂��B
				// ���̒����̐��ۂ��`�F�b�N������ setLastOrder�` ���\�b�h�����s����̂͊댯�ł��B
				om.setLastOrderReason(stockmemo.getReason());
			} else {
				retMsg = "���s";
			}
			
			// �����Ɣ���̕�����؂�ւ��܂��B
			String buysell = "";
			if (stockmemo.isBuy()) {
				buysell = "����";
			} else {
				buysell = "����";
			}
			
			// ���O�o�͗p�̃��b�Z�[�W�����܂��B
			String msg = String.format("�V�K%s�������܂���:%s �����R�[�h:%4d ��������:%,d �������R:%s ��������:%s",
					buysell,
					retMsg,
					stock.getStockCode(),
					qty,
					stockmemo.getReason(),
					om.getLastOrderResult().toString()
					);
			
			// �쐬�������b�Z�[�W�����O�ɏo�͂��܂��B
			RobotLogManager.getInstance().log(msg, 3);
		}
	}
	
	/**
	 * ���Δ����������s���܂��B<BR>
	 * ���̑��v���ɒB�����痘�v�m��܂��͑��؂���s���܂��B
	 * @param profitTakingRate ���v�m�肷�闘�v��(%)�ł��B5.0�Ȃ�A5%�̗��v���ŗ��v�m�肵�܂��B<BR>���̒l(�v���X�̒l)�����Ă��������B
	 * @param lossCutRate      ���؂�����鑹����(%)�ł��B-2.0�Ȃ�A-2%�̗��v��(=2%�̑�����)�ő��؂肵�܂��B<BR>���̒l(�}�C�i�X�̒l)�����Ă��������B
	 * @param maxHoldingDays   �ő�ۗL�����ł��B���̓����𒴂��ĕۗL���Ă����Ȃ炻�̃g���[�h�̗��v���ɂ�����炸���Δ������܂��B
	 */
	protected void orderReverseByRate(double profitTakingRate, double lossCutRate, int maxHoldingDays){
		// �|�[�g�t�H���I�}�l�[�W�����擾���܂��B
		PortfolioManager pm = PortfolioManager.getInstance();
		
		// �|�[�g�t�H���I���Q�Ƃ��āA�K�v������Δ��Δ������s���܂��B
		ArrayList<Portfolio> portfolioList = pm.getPortfolio();
		for (Portfolio portfolio : portfolioList) {
			String reason = "";
			
			// �V�O�i�����o�Ă��Ȃ������ׂ܂��B
			// �V�O�i�����o�Ă���� reason �ϐ��ɗ��R�����܂��B
			double rate = portfolio.getProfitRate();
			if (rate > profitTakingRate || rate < lossCutRate) {
				// ���Δ����̑Ώۂł��B
				
				// ���R��ݒ肵�܂��B
				if (rate > 1) {
					reason = "���v�m��:" + rate;
				} else {
					reason = "���؂�:" + rate;
				}
				
			} else if (maxHoldingDays < portfolio.getHoldingDays()) {
				// �ۗL�����������𒴂���Ɣ��Δ������܂��B
				reason = "�ۗL���������z��";
			}
			
			// �V�O�i�����o�Ă���Δ��Δ������܂��B
			if (0 < reason.length()) {
				orderReverseForPortfolio(portfolio, reason);
			}
		}
	}
	
	/**
	 * ���Δ����������s���܂��B<br>
	 * �V�O�i�����݂āA�V�O�i���Ɣ��΂̃|�W�V����������΂�����������܂��B
	 * @param memo �����I�u�W�F�N�g
	 */
	protected void orderReverseBySignal(SampleObjectMemo memo){
		// �|�[�g�t�H���I�}�l�[�W�����擾���܂��B
		PortfolioManager pm = PortfolioManager.getInstance();
		
		// �|�[�g�t�H���I���Q�Ƃ��āA�K�v������Δ��Δ������s���܂��B
		ArrayList<Portfolio> portfolioList = pm.getPortfolio();
		for (Portfolio portfolio : portfolioList) {
			String reason = "";
			
			// �V�O�i�����o�Ă��Ȃ������ׂ܂��B
			// �V�O�i�����o�Ă���� reason �ϐ��ɗ��R�����܂��B
			if ((portfolio.getExecQty().intValue() < 0) && 
					(memo.contain(true, portfolio.getStock()))) {
				// �󔄂肵�Ă���̂ɔ����V�O�i�����łĂ����甽�Δ��������܂��B
				reason = "�󔄂�ɔ����V�O�i��";
				
			} else if ((portfolio.getExecQty().intValue() > 0) &&
					(memo.contain(false, portfolio.getStock()))) {
				// �����������Ă���̂ɔ���V�O�i�����łĂ����甽�Δ��������܂��B
				reason = "���������ɔ���V�O�i��";
				
			}
			
			// �V�O�i�����o�Ă���Δ��Δ������܂��B
			if (0 < reason.length()) {
				orderReverseForPortfolio(portfolio, reason);
			}
		}
	}
	
	/**
	 * �w��̃|�[�g�t�H���I�ɑ΂��Ĕ��Δ��������{���A���O�Ɏc���܂��B<br>
	 * @param portfolio ���Δ������s�������|�[�g�t�H���I
	 * @param reason    �����̗��R
	 */
	protected void orderReverseForPortfolio(Portfolio portfolio, String reason){
		// �|�[�g�t�H���I���璼�ڔ��Δ����������o���܂��B
		boolean result = portfolio.orderReverseNowMarketAll();
		
		// �����𒍕��̐��ۂŐ؂�ւ��܂��B
		String retMsg = "";
		if(result){
			retMsg = "����";
			
			// �������������Ă����痝�R�����܂��B
			OrderManager.getInstance().setLastOrderReason(reason);
		} else {
			retMsg = "���s";
		}
		
		// ���O�o�͗p�̃��b�Z�[�W�����܂��B
		String msg = String.format(
				"���Δ����������܂���:%s �����R�[�h:%4d ��������:%,d �������R:%s ��������:%s",
				retMsg,
				portfolio.getStock().getStockCode(),
				portfolio.getExecQty().intValue(),
				reason,
				OrderManager.getInstance().getLastOrderResult().toString()
				);
		
		// ���O���o�͂��܂��B
		RobotLogManager.getInstance().log(msg, 3);
	}
	
	@Override
	public void screening(TradeAgent tradeAgent) {
		// ���͎�@�Ǘ��N���X���擾���܂��B
		AnalysisManager analysisManager = tradeAgent.getAnalysisManager();
		
		// �I�u�W�F�N�g������p�ӂ��Anull �`�F�b�N���s���܂��B
		SampleObjectMemo memo = (SampleObjectMemo) MemoManager.getInstance().getObjectMemo();
		if(null == memo) memo = new SampleObjectMemo();
		
		// �����\�胊�X�g��p�ӂ��Anull �`�F�b�N���s���܂��B
		ArrayList<SampleObjectRecord> memoList = memo.getMemoList();
		if(null == memoList) memoList = new ArrayList<SampleObjectRecord>();

		// �Ώۖ����̎擾�����܂�
		ArrayList<Stock> stockList = tradeAgent.getInformationManager().getStockList();
		
		// ���͎�@�̎擾�����܂��B
		// ����RSI�͓����x�[�X��10�񕪂̃f�[�^�𗘗p���镪�͂��s���܂��B
		RSI rsi = analysisManager.getRSI(EnumAnalysisSpan.DAILY, 10);
		MovingAverage ma20 = analysisManager.getMovingAverage(
				EnumAnalysisSpan.DAILY, 20);
		MovingAverage ma40 = analysisManager.getMovingAverage(
				EnumAnalysisSpan.DAILY, 40);
		
		for (Stock stock : stockList) {
			// ���͎�@���\�b�h�N���X����A���ۂ̐��l���o�͂��܂��B
			double rsiVal = rsi.getIndexSimple(stock);
			double maVal20 = ma20.getIndexSimple(stock);
			double maVal40 = ma40.getIndexSimple(stock);
			
			final double rsiBorderBuy  = 25.0;
			final double rsiBorderSell = 90.0;
			
			// - RSI �� 25 �ȉ�
			// - 40���ړ����ς�20���ړ����ς�������Ă���
			// �̗����𖞂����ꍇ�����ƌ��Ȃ��܂��B
			
			// - RSI �� 90 �ȏ�
			// - 20���ړ����ς�40���ړ����ς�������Ă���
			// �̗����𖞂����ꍇ����ƌ��Ȃ��܂��B
			
			if (rsiVal <= rsiBorderBuy && maVal20 > maVal40) {
				// �w���Ώۖ���������������w���\���ǉ����܂��B
				
				// �V�O�i���̋������Z�o���܂��B
				double strength = (rsiBorderBuy - rsiVal) + (maVal20 - maVal40);
				
				// �w���\��ɒǉ�
				memoList.add(new SampleObjectRecord(stock, true, strength, "RSI �����V�O�i������:" + rsiVal));
				
			} else if (rsiVal >= rsiBorderSell && maVal20 < maVal40) {
				// ���p�Ώۖ��������������甄�p�\���ǉ����܂��B
				
				// �V�O�i���̋������Z�o���܂��B
				double strength = (rsiVal - rsiBorderSell) + (maVal40 - maVal20);
				
				// ����\��ɒǉ�
				memoList.add(new SampleObjectRecord(stock, false, strength, "RSI ����V�O�i������:" + rsiVal));
			}
		}
		
		// �V�O�i���̋����Ń\�[�g���܂��B
		memo.sortStockMemoByStrength();
		
		// �I�u�W�F�N�g�����ɃZ�b�g���܂��B
		memo.setMemoList(memoList);
		
		// �I�u�W�F�N�g������o�^���܂��B
		MemoManager.getInstance().setObjectMemo(memo);
	}
	
	/**
	 * �Ώۂ̖������|�[�g�t�H���I�̂Ȃ��ɂ��邩�𒲂ׂ܂��B<br>
	 * @param stock �Ώۖ���
	 * @return �|�[�g�t�H���I�Ɋ܂܂�Ă���(���L���Ă���)�ꍇ true�A�����łȂ��ꍇ(stock��null�̏ꍇ��) false 
	 */
	public boolean isHolding(Stock stock){
		// null �`�F�b�N���s���܂��B
		if (null == stock) return false;
		
		// �����Ώۂ̖����R�[�h���擾���܂��B
		int target_code = stock.getStockCode().intValue();
		
		// �|�[�g�t�H���I���擾���܂��B
		ArrayList<Portfolio> portfolios = PortfolioManager.getInstance().getPortfolio();
		
		// �|�[�g�t�H���I���X�g�̂Ȃ������v������̂�T���܂��B
		for (Portfolio portfolio: portfolios) {
			// �|�[�g�t�H���I�̖����R�[�h���擾����
			int stockcode = portfolio.getStock().getStockCode().intValue();
			
			// �|�[�g�t�H���I�̖����R�[�h�ƁA�����Ώۂ̖����R�[�h����v���邩���ׂ܂��B
			if(stockcode == target_code) return true; // ��v�����珊�����Ă��邱�ƂɂȂ�܂��B
		}
		// ��������܂���ł���
		return false;
	}
}
