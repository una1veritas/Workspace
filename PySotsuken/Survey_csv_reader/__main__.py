'''
Created on 2024/11/25

@author: sin
'''
import math
import statistics

LAB_ASSIGNMENT_UPPERBPUND = 6
DESIREED_LIST_LENGTH = 6

def get_survey_dict(infilename):
    lcounter = 0
    svy_dict = dict()
    with open(infilename,"r") as f:
        for a_line in f:
            lcounter += 1
            if lcounter == 1 : continue
            a_line = a_line.replace('\u3000', ' ')
            items = a_line.strip().split(",")
            lablist = list()
            #print(items[2], items[6:])
            for no_labname in items[6:] :
                if no_labname == '' :
                    continue
                labpair = no_labname.split('：') 
                labpair[0] = int(labpair[0]) 
                if labpair[0] == 23 : 
                    continue
                if len(lablist) > 0 and labpair[0] >= 30 :
                    continue
                lablist.append(labpair)
            #print(lablist)
            if items[2] in svy_dict:
                print('Error: duplicated SID!')
                raise ValueError
            svy_dict[items[2]] = lablist
    return svy_dict
    
def get_gpa_dict(filename = dict()) :
    gpa_dict = dict()
    with open(filename, "r") as f:
        for a_line in f:
            items = a_line.strip().split(',')
            if items[4] == '学生番号' : continue
            if items[4] == '' : continue
            try:
                gpa_dict[items[4]] = float(items[16]) + float(items[17])/1000
            except ValueError:
                gpa_dict[items[4]] = float('NaN')
    return gpa_dict

def assign_by_gpa_first(gpadict, survey_dict):
    gpasorted = sorted(gpadict.items(), key = lambda x: x[1], reverse = True)
    assign = dict()
    sid_set = set([sid for sid, sgpa in gpasorted])
    sum = 0
    count = 0
    for sid, sgpa in gpasorted:
        #print(sid, sgpa)
        if sid not in survey_dict:
            print("Error: No desire found for ",sid, sgpa)
            continue
        desired_list = survey_dict[sid]
        #print(sid, desired)
        for i in range(len(desired_list)) :
            a_pair = desired_list[i]
            (lid, lname) = (a_pair[0], a_pair[1])
            if lid not in assign:
                assign[lid] = list()
            if len(assign[lid]) >= LAB_ASSIGNMENT_UPPERBPUND :
                continue
            assign[lid].append( (sid, math.ceil(sgpa*1000)/1000 ) )
            sum += i + 1
            count += 1
            sid_set.remove(sid)
            break
    print("gpa first satisfaction average = ", sum/count)
    print("remained = ", sid_set)
    return assign

def assign_by_desire_first(gpa_dict, svy_dict):
    gpasorted = sorted(gpa_dict.items(), key = lambda x: x[1], reverse = True)
    set_students = set(gpa_dict.keys())
    labs_determined = set()
    assign = dict()
    sum = 0
    count = 0
    for i in range(DESIREED_LIST_LENGTH) :
        print("desired rank ", i)
        print("labs determined = ", len(labs_determined), "remained = ", len(set_students))
        for sid in set_students:
            if sid in svy_dict and i < len(svy_dict[sid]) :
                lid, lname = svy_dict[sid][i]
                if lid not in assign :
                    assign[lid] = list()
                if len(assign[lid]) < LAB_ASSIGNMENT_UPPERBPUND :
                    assign[lid].append( (sid, math.ceil(gpa_dict[sid]*1000)/1000 ) )
                    sum += i + 1
                    count += 1
        for lid in assign:
            if lid in labs_determined : continue 
            if len(assign[lid]) >= LAB_ASSIGNMENT_UPPERBPUND :
                assign[lid] = assign[lid][:LAB_ASSIGNMENT_UPPERBPUND]
                labs_determined.add(lid)
            for stu in assign[lid] :
                if stu[0] in set_students:
                    set_students.remove(stu[0])
    print("desire first satisfaction average = ", sum/count)
    return assign
    
def write_assignment(assignment, fname):
    with open(fname, "w") as f:
        for lid in sorted(list(assignment.keys())):
            gpa_list = [e[1] for e  in assignment[lid]]
            f.write(str(lid))
            f.write('\t')
            f.write(str(round(statistics.mean(gpa_list), 2)))
            f.write('\t')
            if len(gpa_list) >= 2 :
                f.write(str(round(statistics.stdev(gpa_list), 2)))
            else:
                f.write("--")
            f.write('\t')
            for ea in assignment[lid]:
                f.write(str(ea))
                f.write('\t')
            f.write('\n')
    return
    
if __name__ == '__main__':
    survey_dict = get_survey_dict("./students_svy.csv")
    print(survey_dict)
    gpa_dict = get_gpa_dict("./students_gpa.csv")
    print(gpa_dict)
    assignment = assign_by_gpa_first(gpa_dict, survey_dict)
    write_assignment(assignment, "bygpa.txt")

    assignment = assign_by_desire_first(gpa_dict, survey_dict)
    write_assignment(assignment, "bydesire.txt")
    
    exit(0)
