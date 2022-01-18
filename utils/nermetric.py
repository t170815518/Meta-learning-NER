
import numpy as np

from utils.BIOES_ENCODER import get_TAG_and_pattens


def get_intersection(AG,BSys):
    N = 0
    for b in BSys:
        if b in AG:
            N = N+1
    return N



def get_metric_number_for_each_sentence(g_start, g_end, g_class, sys_start, sys_end, sys_class):


    if len(sys_start) ==0:
        # for total performance
        # G = len(np.where(g_start ==1))  #bug!!!
        G = len(np.where(g_start == 1)[0])
        Re = 0
        Correct = 0

        # for segmentation performance
        SegRe = 0
        SegCorrect = 0
    else:

        g_start_index = np.where(g_start == 1)[0]
        g_end_index = np.where(g_end == 1)[0]

        G = len(g_start_index)

        ZipSegG = list(zip(g_start_index,g_end_index))
        ZipSegSys = list(zip(sys_start,sys_end))

        SegRe = len(ZipSegSys)

        SegCorrect = get_intersection(ZipSegG,ZipSegSys)

        ZigG = list(zip(g_start_index,g_end_index,g_class))
        ZipSys = list(zip(sys_start,sys_end,sys_class))

        Re = len(ZipSys)
        Correct = get_intersection(ZigG,ZipSys)

    return [G,SegRe,SegCorrect,Re,Correct]







def get_metric_number_from_O0B1I2E3S4(seqG,segSys,lens):

    batch_size = len(seqG)


    batch_matrix_number =[]
    for i in range(batch_size):
        cur_L = lens[i]
        cur_g = seqG[i][0:cur_L]
        cur_sys = segSys[i][0:cur_L]

        index_4_sys = np.where(cur_sys.cpu() == 4)[0]
        index_1_sys = np.where(cur_sys.cpu() == 1)[0]

        index_4_g = np.where(cur_g.cpu() == 4)[0]
        index_1_g = np.where(cur_g.cpu() == 1)[0]

        G = len(index_4_g) + len(index_1_g)



        if len(index_4_sys) + len(index_1_sys) == 0:
            Re = 0
            Correct = 0
        else:

            index_3_sys = np.where(cur_sys.cpu() == 3)[0]

            index_3_g = np.where(cur_g.cpu() == 3)[0]

            all_start_g = np.hstack((index_4_g, index_1_g))
            all_end_g = np.hstack((index_4_g, index_3_g))

            all_start_sys = np.hstack((index_4_sys, index_1_sys))
            all_end_sys = np.hstack((index_4_sys, index_3_sys))

            ZigG = list(zip(all_start_g, all_end_g))
            ZipSys = list(zip(all_start_sys, all_end_sys))

            Re = len(ZipSys)
            Correct = get_intersection(ZigG, ZipSys)

        batch_matrix_number.append([G,Re,Correct])


    return batch_matrix_number




def get_ner_seg_score_from_listmatrix(listmatrix):

    np_matrix = np.array(listmatrix)

    G = np.sum(np_matrix[:,0])

    if np.sum(np_matrix[:,1]) !=0:
        SegPrecsion = np.sum(np_matrix[:,2]) / np.sum(np_matrix[:,1])
    else:
        SegPrecsion = 0  # if no return, precison = 0 or =1


    SegRecall = np.sum(np_matrix[:,2]) / G
    if (SegPrecsion+SegRecall)!=0:
        SegF1 = 2* SegPrecsion *SegRecall /(SegPrecsion+SegRecall)
    else:
        SegF1 = 0

    print('Ground:', G)
    print('Total Return:',np.sum(np_matrix[:,1]))
    print('Correct:', np.sum(np_matrix[:,2]))



    return SegPrecsion,SegRecall,SegF1


def get_score_from_listmatrix(listmatrix):

    np_matrix = np.array(listmatrix)

    G = np.sum(np_matrix[:,0])

    if np.sum(np_matrix[:,1]) !=0:
        SegPrecsion = np.sum(np_matrix[:,2]) / np.sum(np_matrix[:,1])
    else:
        SegPrecsion = 0  # if no return, precison = 0 or =1


    SegRecall = np.sum(np_matrix[:,2]) / G
    if (SegPrecsion+SegRecall)!=0:
        SegF1 = 2* SegPrecsion *SegRecall /(SegPrecsion+SegRecall)
    else:
        SegF1 = 0




    if np.sum(np_matrix[:,3])!=0:
        Precison = np.sum(np_matrix[:,4]) / np.sum(np_matrix[:,3])
    else:
        Precison = 0

    Recall = np.sum(np_matrix[:,4]) / G

    if (Precison+Recall)!=0:
        F1 = 2 *Precison *Recall / (Precison+Recall)
    else:
        F1 = 0



    return SegPrecsion,SegRecall,SegF1,Precison,Recall,F1




import re

def get_metric_number_with_RE(seqG, segSys, lens, RE_PATTENS):


    batch_size = len(seqG)

    batch_matrix_number = []

    for i in range(batch_size):

        sys_list_triple = []
        g_list_triple = []


        cur_L = lens[i]

        #GPU
        cur_g = ','.join(map(str,seqG[i][0:cur_L].tolist()))
        cur_sys = ','.join(map(str,segSys[i][0:cur_L].tolist()))

        #CPU
        # cur_g = ','.join(map(str, seqG[i][0:cur_L]))
        # cur_sys = ','.join(map(str, segSys[i][0:cur_L]))

        cur_g_str_to_w = str_index_to_word_index(cur_g)
        cur_sys_str_to_w = str_index_to_word_index((cur_sys))

        #Bug: len(cur_g) != len(cur_sys)

        miter = re.finditer(RE_PATTENS, cur_g)
        for m in miter:
            g_list_triple.append((m.group(), cur_g_str_to_w[m.start()]))

        miter = re.finditer(RE_PATTENS, cur_sys)
        for m in miter:
            sys_list_triple.append((m.group(), cur_sys_str_to_w[m.start()]))



        G = len(g_list_triple)
        Re = len(sys_list_triple)
        Correct = get_intersection(g_list_triple, sys_list_triple)

        batch_matrix_number.append([G, Re, Correct])

    return batch_matrix_number




def str_index_to_word_index(strString):

    str_w_dic = {}

    wordN = 1

    for i, s in enumerate(strString):
        if s == ',':
            wordN = wordN +1
            continue
        else:
            str_w_dic[i] = wordN

    return str_w_dic









if __name__ == '__main__':

    # g_start = np.array([1,0,0,0,0,1,0,1])
    # g_end = np.array([0, 0, 1, 0, 0, 1, 0, 1])
    # g_class = np.array([0,1,3])
    #
    # sys_start = [0,5,7]
    # sys_end = [2,6,7]
    # sys_class = [3,1,3]
    #
    # print(get_metric_number_for_each_sentence(g_start,g_end,g_class,sys_start,sys_end,sys_class))
    #
    # listmatrix= [[3,2,2,2,1],[4,3,2,2,2]]
    #
    # print(get_score_from_listmatrix(listmatrix))

    g =  [[ 0, 4, 0, 5,7,0, 9,10,10,11,0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   9,  11,   0,   0,
           0,   1,   3,   0,   0,   9,  11,   0,   0,   0,   0,   0,   0,   0,
           6,   6,   6]]

    # g = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0,
    #       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #       6, 6, 6]]


    sys =  [[  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   9,  11,   0,   0,
           0,   1,   3,   0,   0,   9,  11,   0,   0,   0,   0,   0,   0,   0,
           6,   6,   6]]
    L = [28]

    TAG_TO_NUNBER, RE_PATTENS = get_TAG_and_pattens('single_conll2003')

    print(get_metric_number_with_RE(g,sys,L, RE_PATTENS))


    input_s = '16,0,0,0,9,11,0,0,0,9,11,0,0,0,9,11,0,0,0,9,11,0,0,0,9,11,0,0,0,9,11,0,0,0,9,11,0,0,0,9,11,0,0,0,0,0,9,11,0,0,0,9,11,0,0,0,9,11,0,0,0,9,11,0,0,0,9,11,0,0,0'

    str_index_to_word_index(input_s)

