package data;

/**
 * @author masayoshi
 *
 * TODO ��Ǝ҂ƍ�Ǝ��Ԃ̑g(����ᔽ���𒲂ׂ鎞���p���Ă���)
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
	 * @return period_id ��߂��܂��B
	 */
	public int getPeriod_id() {
		return period_id;
	}
	/**
	 * @param period_id period_id ��ݒ�B
	 */
	public void setPeriod_id(int period_id) {
		this.period_id = period_id;
	}
	/**
	 * @return processor_id ��߂��܂��B
	 */
	public int getProcessor_id() {
		return processor_id;
	}
	/**
	 * @param processor_id processor_id ��ݒ�B
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
