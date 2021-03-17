FlOPS_BASE = [10.84, 20.87, 50.08, 75.82, 91.29, 102.28, 113.27, 120.84, 131.16, 
              141.48, 151.8, 164.53, 187.23, 209.93, 225.54, 241.02, 256.5, 279.5, 299.57, 300.85]
             
flops_dic = {}
flops_dic['0'] = [32,112,112]
flops_dic['1'] = [16,112,112]
flops_dic['2'] = [24,56,56]
flops_dic['3'] = [24,56,56]
flops_dic['4'] = [32,28,28]
flops_dic['5'] = [32,28,28]
flops_dic['6'] = [32,28,28]
flops_dic['7'] = [64,14,14]
flops_dic['8'] = [64,14,14]
flops_dic['9'] = [64,14,14]
flops_dic['10'] = [64,14,14]
flops_dic['11'] = [96,14,14]
flops_dic['12'] = [96,14,14]
flops_dic['13'] = [96,14,14]
flops_dic['14'] = [160,7,7]
flops_dic['15'] = [160,7,7]
flops_dic['16'] = [160,7,7]
flops_dic['17'] = [320,7,7]
flops_dic['18'] = [1280,7,7]

def conv_flops(k_size, c_in, c_out, h_out, w_out):
    return ((k_size ** 2 * c_in) * c_out + c_out)* h_out * w_out

def count_flops(positions, y_pred_coarse):
    num_class = [3, 2, 3, 2]
    print(len(y_pred_coarse))
    # coarse
    out_coarse = flops_dic[str(positions[0])]
    out_h_coarse = (out_coarse[1] - 1)//2 + 1
    out_w_coarse = (out_coarse[2] - 1)//2 + 1
    print('coarse output size:', out_h_coarse, out_w_coarse)
    convFLOPs_coarse = conv_flops(1, out_coarse[0], 320, out_h_coarse, out_w_coarse) / (10**6)
    print('convFLOPs_coarse:', convFLOPs_coarse)
    linearFLOPs_coarse = 320*num_class[0] / (10**6)
    print('linearFLOPs_coarse:', linearFLOPs_coarse)
    print('FlOPS_BASE:', FlOPS_BASE[positions[0]])
    total_flops = len(y_pred_coarse)* (FlOPS_BASE[positions[0]]+ convFLOPs_coarse + linearFLOPs_coarse)
    print(total_flops/552)

    for i in range(1, len(positions)):
        # check conv for fine classifiers, it will introduce extra flops if its position > coarse position
        if positions[i] > positions[0]:
            total_flops += (Flops_Base[positions[i]] - Flops_Base[positions[0]]) * y_pred_coarse.count(i-1)
        # flops in classifier
        out_fine = flops_dic[str(positions[i])]
        out_h_fine = (out_fine[1] - 1)//2 + 1
        out_w_fine = (out_fine[2] - 1)//2 + 1
        convFLOPs_fine = conv_flops(1, out_fine[0], 320, out_h_fine, out_w_fine)/ (10**6)
        linearFLOPs_fine = 320*num_class[i]/ (10**6)
        total_flops = total_flops + y_pred_coarse.count(i-1)*(convFLOPs_fine + linearFLOPs_fine)
            
    total_flops = total_flops/len(y_pred_coarse)
    print(total_flops)
    return total_flops

