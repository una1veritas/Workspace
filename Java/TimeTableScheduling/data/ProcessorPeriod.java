package data;

/**
 * @author masayoshi
 *
 * TODO 作業者と作業時間の組(制約違反数を調べる時利用している)
 *
 */
public class ProcessorPeriod {
	private int processor_id;
	private int period_id;
	
	public ProcessorPeriod(int processor_id, int period_id){
		this.processor_id = processor_id;
		this.period_id = period_id;
	}
	
	/**
	 * @return period_id を戻します。
	 */
	public int getPeriod_id() {
		return period_id;
	}
	/**
	 * @param period_id period_id を設定。
	 */
	public void setPeriod_id(int period_id) {
		this.period_id = period_id;
	}
	/**
	 * @return processor_id を戻します。
	 */
	public int getProcessor_id() {
		return processor_id;
	}
	/**
	 * @param processor_id processor_id を設定。
	 */
	public void setProcessor_id(int processor_id) {
		this.processor_id = processor_id;
	}
	
	public boolean equals(Object o){
		if(!(o instanceof ProcessorPeriod))
			return false;
		ProcessorPeriod pp = (ProcessorPeriod)o;
		return (pp.processor_id==processor_id)&&(pp.period_id==period_id);
	}
	
	public int hashCode(){
		return processor_id*100+period_id;
	}
}
