from Bio import SeqIO
import numpy as np
import xlsxwriter as xlwt


seq_dict = {rec.id: rec.seq for rec in SeqIO.parse("data/China2021.3.1_numbers.fasta", "fasta")}


#将第一条记录作为参考

reference = seq_dict['1']
#temp = seq_dict.keys()
#print(reference[0])
#print(temp)

lg = len(seq_dict)
gs = 30018


#定义零4x4矩阵，保存变异数量
mutation = np.zeros([4, 4], dtype=float)
#print(mutation)

#其余序列与reference循环比较
#lg为所有序列数量
#gs为单条序列长度
#for i in seq_dict.keys():
#   print(seq_dict[i])

#双层循环遍历字典，统计各个碱基互相转换数量

# f = xlwt.Workbook()#创建工作簿bai
# sheet1 = f.add_worksheet(u'sheet1')#创建sheet
matrix1 = np.zeros([12, lg], dtype=float)
#print(matrix1.shape)

print(len(reference))
for i in seq_dict.keys():
    temp = 0
    temp2 = i
    temp2 = int(temp2) - 1
    # if len(seq_dict[i]) > len(reference):
    for j in seq_dict[i]:
        # print(i)
        # print(temp)
        # print(temp2)
        if temp > len(reference) -1:
            continue
        D1 = seq_dict[i][temp]
        D2 = reference[temp]
        if D2 == 'A' and D1 == 'A':
            mutation[0][0] = mutation[0][0]+1
            temp = temp+1
        if D2 == 'A' and D1 == 'T':
            mutation[0][1] = mutation[0][1]+1
            matrix1[0][temp2] += 1
            temp = temp+1
        if D2 == 'A' and D1 == 'G':
            mutation[0][2] = mutation[0][2]+1
            matrix1[1][temp2] += 1
            temp = temp+1
        if D2 == 'A' and D1 == 'C':
            mutation[0][3] = mutation[0][3]+1
            matrix1[2][temp2] += 1
            temp = temp+1
        if D2 == 'T' and D1 == 'A':
            mutation[1][0] = mutation[1][0]+1
            matrix1[3][temp2] += 1
            temp = temp+1
        if D2 == 'T' and D1 == 'T':
            mutation[1][1] = mutation[1][1]+1
            temp = temp+1
        if D2 == 'T' and D1 == 'G':
            mutation[1][2] = mutation[1][2]+1
            matrix1[4][temp2] += 1
            temp = temp+1
        if D2 == 'T' and D1 == 'C':
            mutation[1][3] = mutation[1][3]+1
            matrix1[5][temp2] += 1
            temp = temp+1
        if D2 == 'G' and D1 == 'A':
            mutation[2][0] = mutation[2][0]+1
            matrix1[6][temp2] += 1
            temp = temp+1
        if D2 == 'G' and D1 == 'T':
            mutation[2][1] = mutation[2][1]+1
            matrix1[7][temp2] += 1
            temp = temp+1
        if D2 == 'G' and D1 == 'G':
            mutation[2][2] = mutation[2][2]+1
            temp = temp+1
        if D2 == 'G' and D1 == 'C':
            mutation[2][3] = mutation[2][3]+1
            matrix1[8][temp2] += 1
            temp = temp+1
        if D2 == 'C' and D1 == 'A':
            mutation[3][0] = mutation[3][0]+1
            matrix1[9][temp2] += 1
            temp = temp+1
        if D2 == 'C' and D1 == 'T':
            mutation[3][1] = mutation[3][1]+1
            matrix1[10][temp2] += 1
            temp = temp+1
        if D2 == 'C' and D1 == 'G':
            mutation[3][2] = mutation[3][2]+1
            matrix1[11][temp2] += 1
            temp = temp+1
        if D2 == 'C' and D1 == 'C':
            mutation[3][3] = mutation[3][3]+1
            temp = temp+1
        if D1 == 'N':
            temp = temp+1

print(mutation)
np.savetxt("MutatuionRate.csv", mutation, delimiter=",")

#计算变异率
EachMutationRate = np.zeros([4, 4], dtype=float)
EachMutationRate = matrix1/(lg*gs)*100

EachMutationRate = np.transpose(EachMutationRate)
print(type(EachMutationRate))

np.savetxt("EachMutationsRate.csv", EachMutationRate, delimiter=",")

