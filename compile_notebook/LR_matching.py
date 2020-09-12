
# import read_ipynb as rd
import re
# from read_ipynb import lis
def getNextLine(lis,line):
    return lis[line+1]

def LR_matching_step(Ltoken,Rtoken,lis,line, isBeizhu):
    temp = lis[line]
  #  print(temp)
   # print("isBeizhu:" + str(isBeizhu))
    if len(temp) > 0:
        for index in range(0, len(temp)):
            if temp[index] == ' ':
                continue

            if temp[index] == '#':
                return 0
            else:
                break
    if isBeizhu == 1:
        return 1
    if (len(re.findall(r'\"\"\"', temp, re.M))%2 == 1 or len(re.findall(r'\"\"\"', temp, re.M))%2 == 1) and isBeizhu == 0:
        return 1
    if (len(re.findall(r'\'\'\'', temp, re.M))%2 == 1 or len(re.findall(r'\'\'\'', temp, re.M))%2 == 1) and isBeizhu == 1:
        return 0
    RPCount = temp.count(Rtoken,0)
    LPCount = temp.count(Ltoken,0)

    #print(temp)
    isInYin = False
    isDanyin = False
    Lcancel = 0
    Rcancel = 0
    for index in range(0,len(temp)):
        if temp[index] == '#':
            break
        if temp[index] == '\"':
            if index != 0:
                if temp[index-1] == '\\':
                    if index-1 > 0:
                        if temp[index-2] != '\\':
                            continue
            if isInYin == False and isDanyin == False:
                isInYin = True
            else:
                isInYin = False
        elif temp[index] == '\'':
            if index != 0:
                if temp[index-1] == '\\':
                    continue
            if isDanyin == False and isInYin == False:
                isDanyin = True
            else:
                isDanyin = False
        elif temp[index] == Ltoken:
            if isInYin == True or isDanyin == True:
                Lcancel += 1
        elif temp[index] == Rtoken:
            if isInYin == True or isDanyin == True:
                Rcancel += 1
   # print(Lcancel)
   # print(Rcancel)

   # LCyin = len(re.findall(r'(\")(.*?)\((.*?)(\")|(\')(.*?)\((.*?)(\')', temp, re.M))
   # RCyin = len(re.findall(r'(\")(.*?)\)(.*?)(\")|(\')(.*?)\)(.*?)(\')', temp, re.M))
   #  LCyin = len(re.findall('(?<=\"\(\":)\"(.+?)\"', temp, re.M))
   #  RCyin = len(re.findall('(?<=\"\)\":)\"(.+?)\"', temp, re.M))
   #  if LCyin > 0:
   #      print("Ltemp:"+temp)
   #      print(LCyin)
   #  if RCyin > 0:
   #      print("Rtemp:"+temp)
   #      print(RCyin)
    RPCount -= Rcancel
    LPCount -= Lcancel


   # c = b.count('CLA')

    Rcount = LPCount - RPCount
    if(Rcount == 0):
        return 0
   # print("####")
    while(Rcount > 0):
        temp = getNextLine(lis, line)
     #   print(temp)
        LPCount = temp.count(Ltoken,0)
        RPCount = temp.count(Rtoken,0)
        loss = RPCount - LPCount

        m = 0
        while m < len(temp):
            if temp[m] is not ' ':
                break
            m = m+1

        temp = temp[m:]
        lis[line] += ' '+temp
        del(lis[line+1])
        Rcount -= loss
    return 0

def Line_feeding(paragraph):
    LFindex = paragraph.find('\n')
    line_set = []
    while(LFindex is not -1):
        line_set.append(paragraph[0:LFindex])
        paragraph = paragraph[LFindex+1:]
        if not paragraph:
            break
        LFindex = paragraph.find('\n')
    if LFindex == -1:
        line_set.append(paragraph)

    return line_set

def Feeding(lis):
    for i in range(0,len(lis)):
        temp = []
        for paragraph in lis[i]["code"]:
            line_set = Line_feeding(paragraph)
            temp += line_set
            lis[i]["code"] = temp
    return lis
        # print("********")

        # for j i n line_set:
        #     print(j)
        # print(line_set)

def LR_matching(Ltoken,Rtoken,lis):
    isBeizhu = 0
    for j in range(0,len(lis)):
        i = 0
        while i <len(lis[j]["code"]):
            isBeizhu = LR_matching_step(Ltoken,Rtoken,lis[j]["code"], i, isBeizhu)
            i = i+1

    return lis
def merge_fanxiegang(lis):
    j = 0
    while (j<len(lis)):
        i = 0
        while (i < len(lis[j]['code'])):
            #print(lis[j]['code'][i])
            if(len(lis[j]['code'][i])) == 0:
                i+=1
                continue
            if(lis[j]["code"][i][-1] == '\\'):
                temp = lis[j]["code"][i+1]
                if len(lis[j]['code'][i+1]) ==0:
                    del (lis[j]["code"][i + 1])
                    i += 1
                    continue
                #   print(temp)
                m = 0
                while m < len(temp):
                    if temp[m] is not ' ':
                        break
                    m = m + 1

                temp = temp[m:]
                # print("temp:" + temp)
                lis[j]["code"][i] += temp
                # print("merged:"+ lis[j]["code"][i])
                del (lis[j]["code"][i+1])
                # print("i:"+str(i))
                # print("len:"+str(len(lis[j]["code"])))
                if lis[j]["code"][i][-1] != '\\':
                    i+=1
                # if i < len(lis[j]["code"]):
                #     print("nextLine:" + lis[j]["code"][i])
            else:
                i+=1
        j+=1

def remove_fanxiegang(lis):
    for j in range(0, len(lis)):
        for i in range(0,len(lis[j]["code"])):
            #print("....")
            if '\\' in lis[j]["code"][i]:
                ind = lis[j]["code"][i].index('\\')
                while ind >= 0:
                    temp =  lis[j]['code'][i][0:ind]
                    temp += lis[j]['code'][i][ind+1:]
                    lis[j]['code'][i] = temp
                    if '\\' in lis[j]["code"][i]:
                        ind = lis[j]["code"][i].index('\\')
                    else:
                        break

def print_lis(lis):
    for j in range(0, len(lis)):
        print("next ********")
        for i in lis[j]["code"]:
            print("....")
            print(i)


def LR_run(lis):
    lis = LR_matching('(',')',lis)
    lis = LR_matching('[',']',lis)
    lis = LR_matching('{','}',lis)

    merge_fanxiegang(lis)
    remove_fanxiegang(lis)
    #print_lis(lis)
    return lis

# lis = run(lis)
# for j in range(0, len(lis)):
#     print("next ********")
#     for i in lis[j]["code"]:
#         print("....")
#         print(i)

#         print(i)
