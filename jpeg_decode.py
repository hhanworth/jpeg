import numpy as np
import cv2
from table import DC_luminance_dict, ac_luminance_dict
import math



# 量化表
Qy = [[16,11,10,16,24,40,51,61],
        [12,12,14,19,26,58,60,55],
        [14,13,16,24,40,57,69,56],
        [14,17,22,29,51,87,80,62],
        [18,22,37,56,68,109,103,92],
        [24,35,55,64,81,104,113,92],
        [49,64,78,87,103,121,120,101],
        [72,92,95,98,112,100,103,99]]
Qc = [[17,18,24,47,99,99,99,99],
        [18,21,26,66,99,99,99,99],
        [24,26,56,99,99,99,99,99],
        [47,66,99,99,99,99,99,99],
        [99,99,99,99,99,99,99,99],
        [99,99,99,99,99,99,99,99],
        [99,99,99,99,99,99,99,99],
        [99,99,99,99,99,99,99,99]]

def read_bits_from_file_with_padding_info(filename):
    # 从文件中读取字节数组
    with open(filename, 'rb') as file:
        byte_array = file.read()

    # 将字节数组中的每个字节转换为比特流
    bits_with_padding_info = ''.join(format(byte, '08b') for byte in byte_array)

    # 提取填充位数信息
    padding_length = int(bits_with_padding_info[:3], 2)

    # 截取正确的数据部分
    bits = bits_with_padding_info[3:-padding_length] if padding_length > 0 else bits_with_padding_info[3:]

    return bits

def bitwise_not_without_sign(num, _n):
    # 获取二进制表示
    binary_representation = bin(num)[2:]
    #print("bitwise",binary_representation.zfill(num))
    # 对每一位取反
    flipped_bits = ''.join(['1' if bit == '0' else '0' for bit in binary_representation.zfill(_n)])

    # 返回按位取反后的结果
    return int(flipped_bits, 2)



filename = './dat/encode_v3.bin'
filename = './dat/test_v3.bin'
read_bits = read_bits_from_file_with_padding_info(filename)

# 打印读取到的比特流
#print(read_bits)



def run(in_bits, Qy_c):
    
    def decode_huffman(bits, huffman_table):
        decoded_message = ''
        current_code = ''

        while bits:
            current_code += bits[0]
            bits = bits[1:]

            # 从字典中查找霍夫曼编码对应的字符
            character = next((char for char, code in huffman_table.items() if code == current_code), None)
            if character is not None:
                return character, bits


    # 读取 h，w
    h_w = in_bits[:32]
    in_bits = in_bits[32:]
    (h, w) = (int(h_w[:16],2),int(h_w[16:32],2))
    print(h,w)

    image_y = np.zeros((h, w), dtype=np.uint8)
    image_u = np.zeros(((h-1)//2+1, (w-1)//2+1), dtype=np.uint8)
    image_v = np.zeros(((h-1)//2+1, (w-1)//2+1), dtype=np.uint8)

    remaining_bits = in_bits
    #print(remaining_bits)
    num_blk = h*w//64
    code_all = []
    last_DC = 0          # 恢复DC值 （差分）
    for j in range(num_blk):
        code = []
        num_finish = 0
        for i in range(64):
            if i ==0:
                #print(remaining_bits[:10])
                decoded_message, remaining_bits = decode_huffman(remaining_bits, DC_luminance_dict)
                #print("1@:",decoded_message)
                ##(remaining_bits[:10])
                if decoded_message == 0:
                    code.append(last_DC)
                    num_finish+=1  
                    remaining_bits = remaining_bits[1:]
                    continue

                #print(type(remaining_bits[0]))
                #print(decoded_message,remaining_bits)
                if remaining_bits[0]!='0':
                    #print(">0",decoded_message, int(remaining_bits[:decoded_message],2), remaining_bits)
                    last_DC += int(remaining_bits[:decoded_message],2)
                    code.append(last_DC)
                    remaining_bits = remaining_bits[decoded_message:]
                    num_finish+=1  
                else:
                    #print("<0",decoded_message)
                    last_DC += -bitwise_not_without_sign(int(remaining_bits[:decoded_message],2),decoded_message)
                    code.append(last_DC)
                    remaining_bits = remaining_bits[decoded_message:]
                    num_finish+=1    
                # print(last_DC)

            else:
                #print(remaining_bits[:30])
                decoded_message, remaining_bits = decode_huffman(remaining_bits, ac_luminance_dict)
                #print("2@:",decoded_message)
                #print(remaining_bits[:10])

                if decoded_message == (0, 0):
                    for k in range(64 - num_finish):
                        code.append(0)
                    #print(remaining_bits)
                    remaining_bits = remaining_bits[1:]
                    #print(remaining_bits)
                    code_all.append(code)
    
                    break
                if decoded_message[1] == 0:
                    for k in range(decoded_message[0]):
                        code.append(0)
                        num_finish+=1 
                    remaining_bits = remaining_bits[1:]
                    continue
                # print(decoded_message,remaining_bits)
                if remaining_bits[0]!='0':
                    #print(decoded_message, int(remaining_bits[:decoded_message[1]],2))
                    for k in range(decoded_message[0]):
                        code.append(0)
                        num_finish+=1
                    code.append(int(remaining_bits[:decoded_message[1]],2))
                    num_finish+=1
                    remaining_bits = remaining_bits[decoded_message[1]:]

                else:
                    #print(int(remaining_bits[:decoded_message[1]],2))
                    #print(decoded_message, -bitwise_not_without_sign(int(remaining_bits[:decoded_message[1]],2),decoded_message[1]))
                    #print("log:",remaining_bits)
                    for k in range(decoded_message[0]):
                        code.append(0)
                        num_finish+=1
                    code.append( -bitwise_not_without_sign(int(remaining_bits[:decoded_message[1]],2),decoded_message[1]))
                    num_finish+=1
                    remaining_bits = remaining_bits[decoded_message[1]:]

    if len(code_all) == 0:
        print("log:未找到结束标志，解码出错！")
     
    #print(code_all)
        

    def inverse_zigzag(zigzag_list):
        block_size = 8
        block = [[0] * block_size for _ in range(block_size)]
        index = 0
        for _s in range(block_size * 2 - 1):
            if _s % 2 == 0:
                for i in range(_s, -1, -1):
                    j = _s - i
                    if i < block_size and j < block_size:
                        block[i][j] = zigzag_list[index]
                        index += 1
            else:
                for j in range(_s, -1, -1):
                    i = _s - j
                    if i < block_size and j < block_size:
                        block[i][j] = zigzag_list[index]
                        index += 1
        return block

    all = []
    for blk in code_all:
        #print(blk)
        block = inverse_zigzag(blk)
        all.append(block)

    #print(all)
    all_new  = []
    for blk in all:
        all_new.append(np.multiply(blk, Qy_c))

    #print(all_new)



    # 反DCT
    def IDCT(dct_list, h, w):
        def IDCT_block(dct_block):
            def alpha(u):
                if u==0:
                    return 1/np.sqrt(8)
                else:
                    return 1/2
        
            block_size = 8
            # dct_block = block_fill(dct_block)
            img_idct = np.zeros((block_size, block_size), dtype=np.float32)
            for x in range(block_size):
                for y in range(block_size):
                    sum_val = 0
                    for u in range(block_size):
                        for v in range(block_size):
                            sum_val += alpha(u) * alpha(v) * dct_block[u, v] * math.cos((2 * x + 1) * u * np.pi / 16) * math.cos((2 * y + 1) * v * np.pi / 16)
                    img_idct[x, y] = sum_val
            img_idct += 128
            return np.clip(np.round(img_idct), 0, 255).astype(np.uint8)
        block_size = 8
        # h, w = image_shape
        img_idct = np.zeros((h, w), dtype=np.uint8)

        k = 0
        for i in range((h + block_size - 1) // block_size):
            for j in range((w + block_size - 1) // block_size):
                dct_block = dct_list[k]
                idct_block = IDCT_block(dct_block)
                img_idct[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] = idct_block
                k += 1
        return img_idct


    restored_image = IDCT(all_new, h,w)
    #print(remaining_bits)
    # print(restored_image)
    return remaining_bits, restored_image


def YUV420_to_RGB(image_y, image_u, image_v):
    h, w = image_y.shape
    rgb_image = np.zeros((h, w, 3), dtype=np.uint8)

    for line in range(h):
        for row in range(w):
            Y = image_y[line, row]
            U = image_u[line // 2, row // 2]
            V = image_v[line // 2, row // 2]

            C = Y - 16
            D = U - 128
            E = V - 128

            R = int(1.164 * C + 1.596 * E)
            G = int(1.164 * C - 0.813 * E - 0.391 * D)
            B = int(1.164 * C + 2.018 * D)

            rgb_image[line, row, 0] = np.clip(R, 0, 255)
            rgb_image[line, row, 1] = np.clip(G, 0, 255)
            rgb_image[line, row, 2] = np.clip(B, 0, 255)

    return rgb_image
'''
432 424
216 216
216 216
'''

rest_bits, y_img = run(read_bits, Qy)    
rest_bits, u_img = run(rest_bits, Qc)  
rest_bits, v_img = run(rest_bits, Qc)  
print(rest_bits)

# img_u_new = np.zeros(u_img.shape, dtype=np.uint8)
# img_v_new = np.zeros(v_img.shape, dtype=np.uint8)

# for i in range(u_img.shpe[0]):
#     for j in range(u_img.shpe[0]):
#         img_u_new[2*i][2*j] = u_img[i][j]
#         img_u_new[2*i+1][2*j] = u_img[i][j]
#         img_u_new[2*i][2*j+1] = u_img[i][j]
#         img_u_new[2*i+1][2*j+1] = u_img[i][j]

#         img_v_new[2*i][2*j] = v_img[i][j]
#         img_v_new[2*i+1][2*j] = v_img[i][j]
#         img_v_new[2*i][2*j+1] = v_img[i][j]
#         img_v_new[2*i+1][2*j+1] = v_img[i][j]
rgb_img = YUV420_to_RGB(y_img, u_img, v_img)

cv2.imshow("img1",y_img)
cv2.imshow("img2",u_img)
cv2.imshow("img3",v_img)
cv2.imshow("img4",rgb_img)
cv2.waitKey(0)