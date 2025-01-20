# 卒業研究配属プログラム
# 作成 尾下 2021.2.15

import csv
import os
import functools

import pandas as pd 

# 入出力ファイル名

# 1. 入力：研究室情報
labs_info_filename = "labs_info-2024.csv"

# 2. 入力：学生成績等情報
students_info_filename = "students_info-2024.csv"

# 3. 入力：学生の配属希望
students_preference_filename = "students_preference-2024.csv"

# 4. 入出力：教員による配属希望理由書にもとづく学生の選択
labs_preference_filename = "labs_preference.csv"

# 5. 出力：各研究室の配属情報
labs_assignments_filename = "labs_assignments.csv"

# 6. 出力：各学生の配属情報
students_assignments_filename = "students_assignments.csv"

# ファイル 1～3 が入力されると、第1希望・成績にもとづく配属のみを行い、4～6 を出力
# ファイル 1～4 が入力されると、全ての配属を行い、5～6 を出力


# 定数
default_num_slots_by_choise = 3  # 配属希望理由書にもとづく配属人数枠
num_slots_for_external_lab = 1  # 学科外研究室の配属定員


# クラス定義
class Student :
	def __init__( self, sid, name, grade ):
		self.sid = sid    # 学生番号
		self.name = name    # 氏名
		self.gpa = float( grade )    # 成績（通計GPA）
		self.intensions = list()    # 第n希望
		# self.lab_no = -1    # 配属研究室（未配属の場合は -1）
		# self.asign_order = -1    # 第何希望で配属されたか（未配属の場合は -1）
		# self.asign_by_choise = False    # 第1希望での配属の場合に、配属希望理由書により配属されたか

class Lab :
	def __init__( self, no, name ):
		self.labid = int( no )    # 研究室番号 self.no
		self.name = name    # 研究室名
		# if int( is_external ) == 0:
		# 	self.is_external = False    # 学科外研究室か
		# else:
		# 	self.is_external = True
		self.capacity = 0    # 定員 self.num_slots 
		self.reserved = 0    # 配属希望理由書による配属定員 self.num_slots_by_choise
		# self.num_open_slots = 0    # 残りの空き定員
		self.applicants = list()    # 第n希望での配属希望者数 self.pref_count
		self.pref_high_grade = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]    # 第n希望での配属希望者の中の最大成績
		# self.priority_order = -1    # 定員を決定する上での研究室の優先順位
		# self.selected_students = list()    # 配属希望理由書にもとづいて教員が選択した学生
		# self.assigned_students = list()    # 研究室に配属された学生
		# self.num_assigned_students = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]    # 第n希望での配属者数
		
		def is_external():
			return self.id < 30


# 研究室同士の比較関数
def com_labs( l0, l1 ):
	# 学科内の研究室が先になるようにソート
	if not l0.is_external and l1.is_external:
		return -1
	if l0.is_external and not l1.is_external:
		return 1
	# 定員が未確定の研究室が先になるようにソート
	if l0.num_slots == 0 and l1.num_slots != 0:
		return -1
	if l0.num_slots != 0 and l1.num_slots == 0:
		return 1
	# 第n希望の学生が多い方が先になるようにソート
	for i in range( 6 ):
		if l0.pref_count[ i ] > l1.pref_count[ i ]:
			return -1
		if l0.pref_count[ i ] < l1.pref_count[ i ]:
			return 1
	# 第n希望の学生の中で最高成績の学生の成績が高い方が先になるようにソート
	for i in range( 6 ):
		if l0.pref_high_grade[ i ] > l1.pref_high_grade[ i ]:
			return -1
		if l0.pref_high_grade[ i ] < l1.pref_high_grade[ i ]:
			return 1
	# 研究室番号でソート
	if l0.no < l1.no:
		return -1
	if l0.no > l1.no:
		return 1
	return 0

# 学生同士の比較関数（成績優先での配属の順序）
def com_students( s0, s1 ):
	# 学生が高い学生が先になるようにソート
	if s0.grade > s1.grade:
		return -1
	if s0.grade < s1.grade:
		return 1
	return 0

# 学生同士の比較関数（教員による配属希望理由書にもとづく学生の選択用の順序）
def com_students2( s0, s1 ):
	# 第1希望の研究室にもとづいてソート（研究室の定員割り当ての優先順位にもとづいてソート）
	if s0.pref[ 0 ] != s1.pref[ 0 ]:
		o0 = lab_index[ s0.pref[ 0 ] ].priority_order
		o1 = lab_index[ s1.pref[ 0 ] ].priority_order
		if o0 < o1:
			return -1
		else:
			return 1
	# 配属済みの学生が未配属の学生よりも先になるようにソート
	if s0.lab_no == s0.pref[ 0 ] and s1.lab_no != s1.pref[ 0 ]:
		return -1
	if s0.lab_no != s0.pref[ 0 ] and s1.lab_no == s1.pref[ 0 ]:
		return 1
	# 成績が高い学生が先になるようにソート
	if s0.grade > s1.grade:
		return -1
	else:
		return 1
	return 0

# 第n希望（成績）による配属
def assign_by_grade( order ):
	for i in lab_index:
		# 研究室の配属可能人数を取得
		lab = lab_index[ i ]
		num_assignments = lab.num_open_slots
		if num_assignments == 0:
			continue
		if order == 0 and num_assignments >= lab.num_slots_by_choise:
			num_assignments -= lab.num_slots_by_choise
		
		# 第n希望で対象研究室を希望している未配属の学生のリストを作成（成績でソート）
		candidates = set()
		for j in student_index:
			student = student_index[ j ]
			if student.lab_no == -1 and student.pref[ order ] == lab.no:
				candidates.add( student )
		sorted_candidates = sorted( candidates, key = functools.cmp_to_key( com_students ) )
		if num_assignments > len( candidates ):
			num_assignments = len( candidates )
		
		# 対象研究室に学生を配属（成績上位の学生を空き定員分を配属）
		for j in range( num_assignments ):
			student = sorted_candidates[ j ]
			student.lab_no = lab.no
			student.asign_order = order + 1
			student.asign_by_choise = False
			lab.assigned_students.add( student.id )
			lab.num_assigned_students[ order ] += 1
		lab.num_open_slots -= num_assignments

# 第1希望（配属希望理由書）による配属
def assign_by_choise():
	order = 0
	for i in lab_index:
		# 研究室の配属可能人数を取得
		lab = lab_index[ i ]
		num_assignments = lab.num_slots_by_choise
		if num_assignments == 0:
			continue
		
		# 第1希望で対象研究室を希望している未配属の学生のリストを作成
		candidates = set()
		for j in student_index:
			student = student_index[ j ]
			if student.lab_no == -1 and student.pref[ order ] == lab.no:
				candidates.add( student )
		
		# 学生を研究室に配属（教員が配属希望理由書にもとづいて選択した学生を配属）
		for j in candidates:
			student = j
			if student.id in lab.selected_students:
				student.lab_no = lab.no
				student.asign_order = order + 1
				student.asign_by_choise = True
				lab.num_assigned_students[ 0 ] += 1
				lab.num_open_slots -= 1
				lab.assigned_students.add( student.id )


if True:  # if this is main executable program
	# ファイル読み込み（研究室情報）
	# 研究室番号, 研究室名, 学科内（0） or 学科外（1）, 固定の定員数
	# labs_info_file = open( labs_info_filename, "r", encoding='utf8' )
	# labs_info_csv = csv.reader( labs_info_file )
	# lab_index = dict()
	# next( labs_info_csv )
	# for i in labs_info_csv:
	# 	lab = Lab( i[ 0 ], i[ 1 ], i[ 2 ] )
	# 	if len( i ) > 3 and len( i[ 3 ] ) > 0:
	# 		lab.num_slots = int( i[ 3 ] )
	# 	lab_index[ lab.no ] = lab
	# labs_info_file.close()
	
	labs_df = pd.read_csv(labs_info_filename, encoding='utf-8-sig',keep_default_na=False, na_filter=False)
	print(labs_df)
	# ファイル読み込み（学生情報）
	# 学生番号, 氏名, 成績（通計GPA）
	# students_info_file = open( students_info_filename, "r", encoding='utf8' )
	# students_info_csv = csv.reader( students_info_file )
	# student_index = dict()
	# next( students_info_csv )
	# for i in students_info_csv:
	# 	student = Student( i[ 0 ], i[ 1 ], i[ 2 ] )
	# 	student_index[ student.id ] = student
	# students_info_file.close()
	students_df = pd.read_csv(students_info_filename, encoding='utf-8-sig',keep_default_na=False, na_filter=False)
	#print(students_df)
	students_dict = dict()
	for _, row in students_df.iterrows() :
		sid = str(row['学生番号'])
		students_dict[sid] = [str(row['氏名']), float(row['成績']), _]
	#print(sorted(students_dict.items()))

	# ファイル読み込み（学生の配属希望）
	# Moodleのアンケートの結果をエクスポートした csv（形式が変わるかもしれないので注意）
	# ユーザーフルネーム, グループ, 学生番号, [メールアドレス], [日付], 第1希望研究室, 第1希望研究室（学科外研究室）, ..., 第6希望研究室, 最終希望研究室（第6希望までで決まらなかった場合のみ追記）
	# students_preference_file = open( students_preference_filename, "r", encoding='utf8' )
	# students_preference_csv = csv.reader( students_preference_file )
	# next( students_preference_csv )
	# missing_students = set()
	# for i in students_preference_csv:
	# 	id = i[ 2 ]
	# 	if id in student_index:
	# 		student = student_index[ id ]
	# 		if len( i[ 6 ] ) > 0: # 学科外研究室の欄に入力がある場合
	# 			student.pref[ 0 ] = int( i[ 6 ][ 0:2 ] )
	# 		else:
	# 			student.pref[ 0 ] = int( i[ 5 ][ 0:2 ] )
	# 		student.pref[ 1 ] = int( i[ 7 ][ 0:2 ] )
	# 		student.pref[ 2 ] = int( i[ 8 ][ 0:2 ] )
	# 		student.pref[ 3 ] = int( i[ 9 ][ 0:2 ] )
	# 		student.pref[ 4 ] = int( i[ 10 ][ 0:2 ] )
	# 		student.pref[ 5 ] = int( i[ 11 ][ 0:2 ] )
	# 		if len( i ) > 12 and len( i[ 12 ] ) > 0:
	# 			student.pref[ 6 ] = int( i[ 12 ][ 0:2 ] )
	# 		else:
	# 			student.pref[ 6 ] = -1
	# 	else:
	# 		missing_students.add( id )
	# students_preference_file.close()

	# 氏名	グループ	学生番号	メールアドレス	日付	第1希望	第2希望	第3希望	第4希望	第5希望	第6希望	第7希望	第8希望	第9希望	第10希望	第11希望	第12希望	第13希望	第14希望	第15希望	第16希望	第17希望	第18希望	第19希望	第20希望
	intentions_df = pd.read_csv(students_preference_filename, encoding='utf-8-sig',keep_default_na=False, na_filter=False)
	# print(intention_df)	
	for _, row in intentions_df.iterrows() :
		sid = str(row['学生番号'])
		intensions = list()
		for labname in row.tolist()[5:] :
			if labname != '' :
				lid, lname = labname.split(u'：')
				lid = int(lid)
				if (lid,lname) not in intensions :
					intensions.append((lid,lname))
		students_dict[sid] += [intensions]
	print(sorted(students_dict.items()))
	exit(1)
	
	nointension_students_ids = set()
	for sid in students_df['学生番号'].unique():
		target_students = intention_df[intention_df['学生番号'] == sid]
		#print(target_students)
		if target_students.empty :
			nointension_students_ids.add(sid)
			print('error: couldn\'t find intensions for ',sid)
		elif len(target_students) > 1 :
			print('error: duplicate students ids in intension list') 
		else:
			intension_list = target_students.iloc[0].tolist()[5:]
			if len(intension_list) == 0 or 'N.A.' in intension_list :
				nointension_students_ids.add(sid)
				print('error: student with empty entry', sid)
		#print(_, targetstu)
	if len(nointension_students_ids) == 0 :
		print('There are no students without intension.')
	
	exit(1)
	# 配属希望が入力されていない学生のリストを作成
	# missing_preference -> nointension_students_ids
 #
 # for i in student_index:
 # 	student = student_index[ i ]
 # 	if student.pref[ 0 ] == -1:
 # 		missing_preference.add( student.id )
 # for i in missing_preference:
 # 	student_index.pop( i )
	
	'''
	# ファイル読み込み（教員による配属希望理由書にもとづく学生の選択）
	# 学生番号, 氏名, 成績（通計GPA）, 第1希望研究室番号, 第1希望研究室名,  第1希望受け入れの可否（◎ 成績による配属決定、○ 希望による配属決定, × 配属不可）, 備考
	available_labs_preference = False
	if os.path.exists( labs_preference_filename ):
		labs_preference_file = open( labs_preference_filename, "r", encoding='utf8' )
		labs_preference_csv = csv.reader( labs_preference_file )
		next( labs_preference_csv )
		for i in labs_preference_csv:
			accept = ( i[ 5 ] == "○" )
			if accept:
				student_id = i[ 0 ]
				lab_no = int( i[ 3 ] )
				lab = lab_index[ lab_no ]
				lab.selected_students.add( student_id )
		labs_preference_file.close()
		available_labs_preference = True
	
	# 不足情報の表示
	if len( missing_students ) > 0:
		print( "次の学生の成績の情報が未入力: " )
		print( missing_students )
	if len( missing_preference ) > 0:
		print( "次の学生の配属希望が未入力: " )
		print( missing_preference )
	'''
	
	# 研究室の基本定員数を計算
	num_internal_labs = 0  # 標準的な学科内研究室数
	num_external_labs = 0  # 学生が配属される学科外研究室の数
	num_fixed_slots = 0  # 標準的な学科内研究室以外の配属人数
	for i in lab_index:
		lab = lab_index[ i ]
		if not lab.is_external:
			if lab.num_slots > 0:
				num_fixed_slots += lab.num_slots
			else:
				num_internal_labs += 1
		else:
			for j in student_index:
				student = student_index[ j ]
				if student.pref[ 0 ] == lab.no:
					num_external_labs += 1
					num_fixed_slots += num_slots_for_external_lab
					break
	num_internal_students = len( student_index ) - num_fixed_slots
	lower_slots = num_internal_students // num_internal_labs
	upper_slots = lower_slots + 1
	num_upper_slots_labs = num_internal_students - num_internal_labs * lower_slots
	
	# 各研究室の志望者数のカウント＋最高成績の情報を設定
	for i in lab_index:
		lab = lab_index[ i ]
		for j in student_index:
			student = student_index[ j ]
			for k in range( 7 ):
				if student.pref[ k ] == lab.no:
					lab.pref_count[ k ] += 1
					if student.grade > lab.pref_high_grade[ k ]:
						lab.pref_high_grade[ k ] = student.grade;
	
	# 全研究室を優先順位でソート
	lab_sorted = sorted( lab_index.values(), key = functools.cmp_to_key( com_labs ) )
	
	# 優先順位に従って、各研究室の定員を決定
	for i in range( len( lab_sorted ) ):
		# 研究室の優先順位を記録
		lab = lab_sorted[ i ]
		lab.priority_order = i
		
		# 定員が確定している研究室はスキップ
		if lab.num_slots > 0:
			lab.num_slots_by_choise = 0
			lab.num_open_slots = lab.num_slots
			continue
	
		# 学科内研究室の定員（優先順位が高い研究室は定員数を+1する）
		if i < num_upper_slots_labs:
			lab.num_slots = upper_slots
		else:
			if not lab.is_external:
				lab.num_slots = lower_slots
			else: 
				# 学科外研究室の定員（希望者がいない研究室は 0名とする。）
				if lab.pref_count[ 0 ] >= num_slots_for_external_lab:
					lab.num_slots = num_slots_for_external_lab
				else:
					if lab.pref_count[ 0 ] > 0:
						lab.num_slots = lab.pref_count[ 0 ]
					else:
						lab.num_slots = 0
		# 配属希望理由書にもとづく配属人数枠
		if lab.pref_count[ 0 ] > lab.num_slots:
			if not lab.is_external:
				lab.num_slots_by_choise = default_num_slots_by_choise
			else:
				lab.num_slots_by_choise = 1
		else:
			lab.num_slots_by_choise = 0
		
		# 空き定員数を初期化
		lab.num_open_slots = lab.num_slots
	
	

	# 全学生を全研究室に順番に配属
	# 教員による配属希望理由書にもとづく学生の選択の情報が入力されていなければ、第1希望・成績にもとづく配属までを実行して終了
	assign_by_grade( 0 )
	if available_labs_preference:
		assign_by_choise()
		for i in range( 1, 7 ):
			assign_by_grade( i )
	
	
	# 配属結果を出力（各研究室の配属情報）
	output = open( labs_assignments_filename, "w", encoding='utf_8_sig' )
	output.write( "研究室番号, 研究室名, 定員, 残りの配属可能人数, " )
	output.write( "第1希望配属者数（成績）, 第1希望配属者数（配属希望理由書）, 第2希望配属者数, 第3希望配属者数, 第4希望配属者数, 第5希望配属者数, 第6希望配属者数, 追加希望配属者数, " )
	output.write( "第1希望者数, 第2希望者数, 第3希望者数, 第4希望者数, 第5希望者数, 第6希望者数, 追加希望者数\n" )
	for i in lab_index:
		lab = lab_index[ i ]
		output.write( format( lab.no, "0>2d" )+ "," )
		output.write( lab.name + "," )
		output.write( str( lab.num_slots ) + "," )
		output.write( str( lab.num_open_slots ) + "," )
		if lab.num_open_slots == 0:
			output.write( str( lab.num_assigned_students[ 0 ] - lab.num_slots_by_choise ) + "," )
			output.write( format( int( lab.num_slots_by_choise ), "d" ) + "," )
		else:
			output.write( str( lab.num_assigned_students[ 0 ] ) + "," )
			output.write( "0," )
		for j in range( 1, 7 ):
			output.write( format( int( lab.num_assigned_students[ j ] ), "d" ) + "," )
		for j in range( 0, 6 ):
			output.write( format( int( lab.pref_count[ j ] ), "d" ) + "," )
		output.write( format( int( lab.pref_count[ 6 ] ), "d" ) )
		output.write( "\n" )
	output.close()
	
	# 配属結果を出力（各学生の配属情報）
	output = open( students_assignments_filename, "w", encoding='utf_8_sig' )
	output.write( "学生番号, 氏名, 配属研究室番号, 配属研究室名, 希望順位（1～6 or NA）, 配属方法（成績 or 希望理由書）\n" )
	for i in student_index:
		student = student_index[ i ];
		if lab_index.get( student.lab_no ):
			lab = lab_index[ student.lab_no ]
		else:
			lab = None
		output.write( student.id + "," )
		output.write( student.name + "," )
		output.write( format( student.lab_no, "0>2d" ) + "," )
		if ( lab ):
			output.write( lab.name + "," )
		else:
			output.write( "未定" + "," )
		output.write( str( student.asign_order ) + "," )
		if ( lab ):
			if ( student.asign_by_choise ):
				output.write( "希望理由書" + "\n" )
			else:
				output.write( "成績" + "\n" )
		else:
			output.write( "未定" + "\n" )
	output.close()
	
	# 教員による配属希望理由書にもとづく学生の選択のための中間結果を出力（情報が未入力の場合のみ出力）
	if not available_labs_preference:
		sorted_students = sorted( student_index.values(), key = functools.cmp_to_key( com_students2 ) )
	
		output = open( labs_preference_filename, "w", encoding='utf_8_sig' )
		output.write( "学生番号, 氏名, 成績（通計GPA）, 第1希望研究室番号, 第1希望研究室名, 第1希望受け入れの可否（◎ 成績による配属決定、○ 希望による配属決定、× 配属不可）, 備考\n" )
		for student in sorted_students:
			if lab_index.get( student.pref[ 0 ] ):
				lab = lab_index[ student.pref[ 0 ] ]
			else:
				lab = None
			output.write( student.id + "," )
			output.write( student.name + "," )
			output.write( str( student.grade ) + "," )
			output.write( format( lab.no, "0>2d" ) + "," )
			output.write( lab.name + "," )
			if ( student.lab_no > 0 ):
				output.write( "◎,\n" )
			else:
				if lab.num_slots_by_choise > 0:
					output.write( "○ or ×," )
					output.write( str( lab.num_open_slots ) + "名を選択して○を記入してください。\n" )
				else:
					output.write( ",\n" )
		output.close()
