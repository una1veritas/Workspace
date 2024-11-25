'''
Created on 2024/11/25

@author: sin
'''

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
            for no_labname in items[6:] :
                labpair = no_labname.split('：')
                if labpair[0] != '' :
                    labpair[0] = int(labpair[0])
                    if labpair[0] == 23 : continue
                    lablist.append(labpair)
            if items[2] in svy_dict:
                print('Error: duplicated SID!')
                raise ValueError
            if lablist[0] == 23 :
                lablist = lablist[2:]
            else:
                lablist = lablist[:2] + lablist[4:]  
            print(lablist)
            svy_dict[items[2]] = lablist
    return svy_dict
    
def get_gpa_dict(filename = dict()) :
    gpa_dict = dict()
    with open(filename, "r") as f:
        for a_line in f:
            items = a_line.strip().split(',')
            if items[4] == '学生番号' : continue
            try:
                gpa_dict[items[4]] = float(items[16]) + float(items[17])/1000
            except ValueError:
                gpa_dict[items[4]] = float('NaN')
    return gpa_dict

if __name__ == '__main__':
    survey_dict = get_survey_dict("./students_svy.csv")
    print(survey_dict)
    gpa_dict = get_gpa_dict("./students_gpa.csv")
    print(gpa_dict)
    gpasorted = sorted(gpa_dict.items(), key = lambda x: x[1], reverse = True)
    print(gpasorted)
    assign = dict()
    count = 0
    xcount = 0
    sid_set = set([sid for sid, sgpa in gpasorted])
    for sid, sgpa in gpasorted:
        print(sid, sgpa)
        if sid not in survey_dict:
            print("Error: No desire found for " + sid)
            continue
        desired = survey_dict[sid]
        print(sid, desired)
        for a_pair in desired:
            (lid, lname) = (a_pair[0], a_pair[1])
            print(lid, lname)
            if lid not in assign:
                assign[lid] = list()
            if len(assign[lid]) >= 5 :
                xcount += 1
                continue
            assign[lid].append(sid)
            sid_set.remove(sid)
            count += 1
            break
    sum = 0
    for lid, assigned in assign.items():
        print(lid, assigned)
        sum += len(assigned)
    print("count=", count, " xcount=", xcount)
    print("sum = ", sum, " sid_set size ", len(sid_set))
    exit(0)
