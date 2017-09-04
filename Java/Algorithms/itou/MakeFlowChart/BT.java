public class BT {
	// �񕪖؂̓��ł��D���̉E�̎q�ߓ_���񕪖؂̍��ɂȂ�܂��D
	private BTNode head;

	// ��̓񕪖؂��\�z���܂��D
	public BT() {  this(null);  }

	// �����w�肵�āC�񕪖؂��\�z���܂��D
	// @param root �񕪖؂̍�
	public BT(BTNode root) {
		head = new BTNode(); // �؂̓��𐶐����܂��D
		head.setZero(root); // ���̍������[�g�ɂ��܂��D
	}

	// �؂̍���ݒ肵�܂�
	// @param node ���m�[�h
	public void setRoot(BTNode node) { head.setZero(node); }

	// �؂̍����擾���܂�
	// @return ���m�[�h
	public BTNode getRoot() { return head.getZero(); }

	/**
	 * �m�[�h���؂̍����ǂ����𔻒f���܂��D
	 * @param node �m�[�h
	 * @return ���Ȃ�^
	 */
	public boolean isRoot(BTNode node) { return getRoot() == node; }
	
	/**
	 * �؂̓����擾���܂��D�擾����̂݁D
	 * @return �؂̓�
	 */
	protected BTNode getHead() { return head; }

	/**
	 * �񕪖؂�\�����܂��D�C���I�[�_�ŏo�͂��C�C���f���g�����܂��D
	 * ���ۂ̏����͕⏕������ċA�I�Ɏg�p���܂��D
	 */
	public void show() { show(getRoot(), 0); }

	/**
	 * �񕪖؂�\�����邽�߂̍ċA�I�����ł��D
	 * @param node �\������ߓ_
	 * @param level �C���f���g�̐[��
	 */
	private void show(BTNode node, int level) {
		System.out.println(node);
		if(node.getZero() != null){
			for (int i = 0; i <= level; i++)
				System.out.print("       ");
			System.out.print("�k");
			show(node.getZero(), level + 1);
		}
		int childNUM=0;
		childNUM = node.getChildNUM();
		for(int j=1; j<childNUM; j++){
			if(node.getChild(j) != null){
				for (int i = 0; i <= level; i++)
					System.out.print("       ");
				System.out.print("�k");
				show(node.getChild(j), level + 1);
			}
		}
	}
}
