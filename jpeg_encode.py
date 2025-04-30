import numpy as np
import cv2
from table import DC_luminance_dict, ac_luminance_dict


# nblock = 0
# now = 0

# # 色彩空间转换 BGR -> YUV 4:2:0
# def RGB2YUV420(image):

#     h, w, c = image.shape
#     #print(h,w)
#     # 转换图片大小，必须能被切分成8*8的小块
#     if((h % 8 == 0) and (w % 8 == 0)):
#         nblock = h * w // 64
#     else:
#         h = h // 8 * 8 + 8 
#         w = w // 8 * 8 + 8
#         image = cv2.resize(image, [w, h], cv2.INTER_CUBIC)
#         print(image.shape)
#         # nblock = h * w // 64
#     #h, w, c = image.shape
#     #print(h,w)

#     image_y = np.zeros((h, w), dtype=np.uint8)
#     image_u = np.zeros(((h-1)//2+1, (w-1)//2+1), dtype=np.uint8)
#     image_v = np.zeros(((h-1)//2+1, (w-1)//2+1), dtype=np.uint8)
#     for line in range(h):
#         for row in range(w):
#             B = image[line, row, 0]
#             G = image[line, row, 1]
#             R = image[line, row, 2]
#             Y = np.round(0.299*R + 0.587*G + 0.114*B)
#             image_y[line, row] = Y
#             if line % 2 == 0 and row % 2 == 0:
#                 U = np.round(0.5*R - 0.4187*G - 0.0813*G + 128)
#                 image_u[line//2, row//2] = U 
#             if line % 2 == 1 and row %2 == 1:
#                 V = np.round(-0.1687*R - 0.3313*G + 0.5*B + 128)
#                 image_v[line//2, row//2] = V

#     return image_y, image_u, image_v


def RGB2YUV420(image):
    # 使得到的图像可以正好分成 8*8 的子块
    def adjust_size(image):
        h, w  = image.shape[0], image.shape[0]
        new_h = h // 8 * 8
        new_w = w // 8 * 8                          # 三次样条插值
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)    
        return image
    image = adjust_size(image)
    h, w, c = image.shape
    image_y = np.zeros((h, w), dtype=np.uint8)
    image_u = np.zeros((h // 2, w // 2), dtype=np.uint8)
    image_v = np.zeros((h // 2, w // 2), dtype=np.uint8)
    for line in range(h):
        for row in range(w):
            B = image[line, row, 0]
            G = image[line, row, 1]
            R = image[line, row, 2]
            Y = np.round(0.299 * R + 0.587 * G + 0.114 * B)
            image_y[line, row] = Y
            if line % 2 == 0 and row % 2 == 0:
                U = np.round(0.5 * R - 0.4187 * G - 0.0813 * B + 128)
                image_u[line // 2, row // 2] = U
            if line % 2 == 1 and row % 2 == 1:
                V = np.round(-0.1687 * R - 0.3313 * G + 0.5 * B + 128)
                image_v[line // 2, row // 2] = V
    image_y = adjust_size(image_y)
    image_u = adjust_size(image_u)
    image_v = adjust_size(image_v)
    return image_y, image_u, image_v


import math



    
def DCT(image):
    def alpha(u):
        if u==0:
            return 1/np.sqrt(8)
        else:
            return 1/2
    def DCT_block(img):
        block_size = 8
        img_fp32 = img.astype(np.float32)
        img_fp32 -= 128
        img_dct = np.zeros((block_size, block_size), dtype=np.float32)
        for line in range(block_size):
            for row in range(block_size):
                n = 0
                for x in range(block_size):
                    for y in range(block_size):
                        n += img_fp32[x,y]*math.cos(line*np.pi*(2*x+1)/16)*math.cos(row*np.pi*(2*y+1)/16)
                img_dct[line, row] = alpha(line)*alpha(row)*n
        return np.ceil(img_dct)
    
    block_size = 8
    h, w = image.shape
    dlist = []
    for i in range((h + block_size - 1) // block_size):
        for j in range((w + block_size - 1) // block_size):
            img_block = image[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
            # 处理一个像素块
            img_dct = DCT_block(img_block)
            dlist.append(img_dct)
    return dlist



def quantization(blocks, Q):
    img_quan = []
    for block in blocks:                # divide --> 数组对应位置做除法
        img_quan.append(np.round(np.divide(block, Q)))
    return img_quan


def zigzag(blocks):
    block_list = []
    for block in blocks:
        zlist = []
        w, h = block.shape
        if w != h:
            return None
        max_sum = w + h - 2
        for _s in range(max_sum + 1):
            if _s % 2 == 0:
                for i in range(_s, -1, -1):
                    j = _s - i
                    if i >= w or j >= h:
                        continue
                    zlist.append(block[i,j])
            else:
                for j in range(_s, -1, -1):
                    i = _s - j
                    if i >= w or j >= h:
                        continue
                    zlist.append(block[i,j])
        block_list.append(zlist)
    return block_list

def DPCM(zglist):
    res_dpcm = []
    for i in range(len(zglist)):
        if i == 0:
            res_dpcm.append(zglist[i][0])
            continue
        res_dpcm.append(zglist[i][0]-zglist[i-1][0])
    return res_dpcm

def rlc(zglist):
    res_ac = []
    for i in range(len(zglist)):
        ac = []
        zg = zglist[i]
        zero_num = 0
        for k in range(1, len(zg)):
            if zg[k] != 0:
                ac.append((zero_num, zg[k]))
                zero_num = 0
            else:
                zero_num += 1
                if zero_num == 16:
                    ac.append((zero_num-1, 0))
                    zero_num = 0
        if zero_num:
            ac.append((zero_num-1, 0))
        
        # 检查最后部分是否全0
        _num = 0
        for dat in reversed(ac):
            if dat[1] == 0:
                _num+=1
        for i in range(_num):
            ac.pop()
        ac.append((0,0))
        res_ac.append(ac)

    return res_ac


if __name__ == '__main__': 

    image = cv2.imread('./pic/p.bmp')

    #  BGR -> YUV
    # image_yuv = np.zeros_like(image, dtype=np.uint8)
    # for line in range(h):
    #     for row in range(w):
    #         B = image[line, row, 0]
    #         G = image[line, row, 1]
    #         R = image[line, row, 2]
    #         Y = np.round(0.299*R + 0.587*G + 0.114*B)
    #         U = np.round(0.5*R - 0.4187*G - 0.0813*G + 128)
    #         V = np.round(-0.1687*R - 0.3313*G + 0.5*B + 128)
    #         image_yuv[line, row, :] = (Y, U, V)    
    
    #  4:2:0 进行 dct
    image_y, image_u, image_v = RGB2YUV420(image)
    #print(image_y)

    y_hight, y_width= image_y.shape
    u_hight, u_width= image_u.shape
    v_hight, v_width= image_v.shape
    # 保存图像
    # cv2.imwrite('./pic/Y.png', image_y)
    # cv2.imwrite('./pic/U.png', image_u)
    # cv2.imwrite('./pic/V.png', image_v)     
    #cv2.imwrite('./pic/YUV.png', image_yuv) 
    
    # img_dct = DCT(np.array([[203,191,195,211,191,174,203,196],
    #                         [207,198,202,211,194,177,203,197],
    #                         [211,211,210,209,197,182,203,198],
    #                         [212,214,216,208,198,184,201,197],
    #                         [213,215,216,209,198,182,202,208],
    #                         [217,218,218,209,198,188,208,212],
    #                         [221,222,222,211,202,196,214,216],
    #                         [223,224,226,217,210,205,217,217],

    #                         [196,191,195,211,191,174,203,196],
    #                         [207,286,202,211,194,153,203,197],
    #                         [211,211,210,209,197,197,203,153],
    #                         [212,188,216,164,198,184,201,197],
    #                         [157,215,216,209,198,182,202,208],
    #                         [183,218,196,188,198,188,208,212],
    #                         [221,222,222,211,188,196,196,216],
    #                         [223,224,226,224,149,157,157,217]]))
    
    # y
    img_ydct = DCT(image_y)
    img_udct = DCT(image_u)
    img_vdct = DCT(image_v)
    #print(np.array(img_dct).shape)
    # exit()
    # u
    #img_udct = DCT(image_u)
    # v
    #img_vdct = DCT(image_v)    
    # with open('./dat/src.txt', 'w') as f:  
    #     f.writelines(str(image_y))
    #     f.close()
    # with open('./dat/dct.txt', 'w') as f:  
    #     f.write(str(img_ydct))
    #     f.close()
    # exit()

    zglist_y = zigzag(img_ydct)
    zglist_u = zigzag(img_udct)
    zglist_v = zigzag(img_vdct)
    # 量化表
    # Qy = [[16,11,10,16,24,40,51,61],
    #       [12,12,14,19,26,58,60,55],
    #       [14,13,16,24,40,57,69,56],
    #       [14,17,22,29,51,87,80,62],
    #       [18,22,37,56,68,109,103,92],
    #       [24,35,55,64,81,104,113,92],
    #       [49,64,78,87,103,121,120,101],
    #       [72,92,95,98,112,100,103,99]]
    # Qc = [[17,18,24,47,99,99,99,99],
    #       [18,21,26,66,99,99,99,99],
    #       [24,26,56,99,99,99,99,99],
    #       [47,66,99,99,99,99,99,99],
    #       [99,99,99,99,99,99,99,99],
    #       [99,99,99,99,99,99,99,99],
    #       [99,99,99,99,99,99,99,99],
    #       [99,99,99,99,99,99,99,99]]
    Qy = [16,11,10,16,24,40,51,61,
          12,12,14,19,26,58,60,55,
          14,13,16,24,40,57,69,56,
          14,17,22,29,51,87,80,62,
          18,22,37,56,68,109,103,92,
          24,35,55,64,81,104,113,92,
          49,64,78,87,103,121,120,101,
          72,92,95,98,112,100,103,99]
    Qc = [17,18,24,47,99,99,99,99,
          18,21,26,66,99,99,99,99,
          24,26,56,99,99,99,99,99,
          47,66,99,99,99,99,99,99,
          99,99,99,99,99,99,99,99,
          99,99,99,99,99,99,99,99,
          99,99,99,99,99,99,99,99,
          99,99,99,99,99,99,99,99]
    # 量化
    imgy_quan = quantization(zglist_y, Qy)
    imgu_quan = quantization(zglist_u, Qc)
    imgv_quan = quantization(zglist_v, Qc)
    #print(img_quan)


    #print(zglist)
    # 直流系数
    res_dpcm_y = DPCM(imgy_quan)
    res_dpcm_u = DPCM(imgu_quan)
    res_dpcm_v = DPCM(imgv_quan)
    #print(res_dpcm)

    # AC系数
    res_ac_y = rlc(imgy_quan)
    res_ac_u = rlc(imgu_quan)
    res_ac_v = rlc(imgv_quan)
    #print(res_ac)  
    
    # 化为中间格式
    def convert_to_std_format(res_dpcm, res_ac):
        blk_all = []
        for i in range(len(res_ac)):
            blk = []
            for j in range(len(res_ac[i])+1):
                if j==0:
                    blk.append([(int(res_dpcm[i]).bit_length()),(int(res_dpcm[i]))])
                else:
                    blk.append([(res_ac[i][j-1][0],int(abs(res_ac[i][j-1][1])).bit_length()),(res_ac[i][j-1][1])])
            blk_all.append(blk)
        return blk_all
    blk_all_y = convert_to_std_format(res_dpcm_y, res_ac_y)
    blk_all_u = convert_to_std_format(res_dpcm_u, res_ac_u)
    blk_all_v = convert_to_std_format(res_dpcm_v, res_ac_v)


    #print(blk_all)    
    def code_by_huffman(blk_all,hight, width):
            # 二进制反转
        def bitwise_not_without_sign(num):
            # 获取二进制表示
            binary_representation = bin(num)[2:]
            # 对每一位取反
            flipped_bits = ''.join(['1' if bit == '0' else '0' for bit in binary_representation])
            # 返回按位取反后的结果
            return int(flipped_bits, 2)
        # 熵编码 根据 表格
        # 对于DC系数的中间格式(2)(3)而言，数字2查DC亮度Huffman表得到011，数字3通过查找VLI编码表得到其被编码为11；
        # 对于AC系数的中间格式(1,2)(-2)而言，(1,2)查AC亮度Huffman表得到11011，-2通过查找VLI编码表得到其被编码为01；
        code_ret = ''
        # 使用 16 位保存高度和宽度的信息
        height_bits = bin(hight)[2:].zfill(16)
        width_bits = bin(width)[2:].zfill(16)
        code_ret += height_bits + width_bits            # 32位
        for i in range(len(blk_all)):
            for j in range(len(blk_all[i])):
                if j==0:
                    if blk_all[i][j][1] >= 0 :
                        code_ret+= DC_luminance_dict.get(blk_all[i][j][0])
                        code_ret+= str(bin(blk_all[i][j][1])[2:])
                    else:
                        code_ret+= DC_luminance_dict.get(blk_all[i][j][0])
                        code_ret+= str(bin(bitwise_not_without_sign(int(abs(blk_all[i][j][1]))))[2:]).zfill(blk_all[i][j][0])
                else:
                    if blk_all[i][j][1] >= 0 :
                        code_ret+= str(ac_luminance_dict.get(blk_all[i][j][0])) + str(bin(int(blk_all[i][j][1]))[2:])
                    else:
                        code_ret+= str(ac_luminance_dict.get(blk_all[i][j][0])) +  str(bin(bitwise_not_without_sign(int(abs(blk_all[i][j][1]))))[2:]).zfill(blk_all[i][j][0][1])
        return code_ret
        
            #print(ret)
        #print(code_ret)

    huffman_all_y = code_by_huffman(blk_all_y, y_hight, y_width)
    huffman_all_u = code_by_huffman(blk_all_u, u_hight, u_width)
    huffman_all_v = code_by_huffman(blk_all_v, v_hight, v_width)

    huffman_all = huffman_all_y + huffman_all_u + huffman_all_v
    def write_bits_to_file_with_padding_info(bits, filename):
        # 计算需要填充的零的数量
        padding_length = (8 - (len(bits) + 3) % 8) % 8
        # 在比特流的末尾添加零
        bits += '0' * padding_length
        # 将填充的位数信息附加到比特流中
        bits_with_padding_info = format(padding_length, '03b') + bits
        # 将比特流按每8位分割成字节
        bytes_data = [bits_with_padding_info[i:i+8] for i in range(0, len(bits_with_padding_info), 8)]
        # 将每个字节转换为整数
        byte_values = [int(byte, 2) for byte in bytes_data]
        # 将整数列表转换为字节数组
        byte_array = bytearray(byte_values)
        # 写入字节数组到文件    附加方式
        with open(filename, 'wb') as file:
            file.write(byte_array)
     
    write_bits_to_file_with_padding_info(huffman_all, './dat/test_v4.bin')











