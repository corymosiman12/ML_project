import numpy as np

def form_features(coarse_vec_field):
    if isinstance(coarse_vec_field, list):
        new_list = []
        for vec_field in coarse_vec_field:
            new_list.append(forming_features(vec_field))
        return new_list
    elif isinstance(coarse_vec_field, dict):
        return forming_features(coarse_vec_field)


def forming_features(coarse_vec_field, train = False):
    new_dict = {}
    total_count = 0
    for direction, vel_array in coarse_vec_field.items():
        feature_dict = {}
        for ii in range(vel_array.shape[0]):
            if not ii in feature_dict.keys():
                feature_dict[ii] = {}
            for jj in range(vel_array.shape[1]):
                if ii >= 1 and jj >=1 and ii <=254 and jj <= 254:
                    # middle rows
                    total_count += 1
                    a1 = vel_array[ii-1][jj-1]
                    a2 = vel_array[ii-1][jj]
                    a3 = vel_array[ii-1][jj+1]
                    a4 = vel_array[ii][jj-1]
                    a5 = vel_array[ii][jj]
                    a6 = vel_array[ii][jj+1]
                    a7 = vel_array[ii+1][jj-1]
                    a8 = vel_array[ii+1][jj]
                    a9 = vel_array[ii+1][jj+1]

                    feature_dict[ii][jj] = [a1, a2, a3, a4, a5, a6, a7, a8, a9]

                elif ii == 0 and jj == 0:
                    # top left
                    total_count += 1
                    a1 = vel_array[255][255]
                    a2 = vel_array[255][1]
                    a3 = vel_array[255][2]
                    a4 = vel_array[0][255]
                    a5 = vel_array[0][0]
                    a6 = vel_array[0][1]
                    a7 = vel_array[1][255]
                    a8 = vel_array[1][0]
                    a9 = vel_array[1][1]

                    feature_dict[ii][jj] = [a1, a2, a3, a4, a5, a6, a7, a8, a9]

                elif ii == 255 and jj == 0:
                    # bottom left
                    total_count += 1
                    a1 = vel_array[254][255]
                    a2 = vel_array[254][0]
                    a3 = vel_array[254][1]
                    a4 = vel_array[255][255]
                    a5 = vel_array[255][0]
                    a6 = vel_array[255][1]
                    a7 = vel_array[0][255]
                    a8 = vel_array[0][0]
                    a9 = vel_array[0][1]

                    feature_dict[ii][jj] = [a1, a2, a3, a4, a5, a6, a7, a8, a9]

                elif ii == 0 and jj == 255:
                    # top right
                    total_count += 1
                    a1 = vel_array[255][254]
                    a2 = vel_array[255][255]
                    a3 = vel_array[255][0]
                    a4 = vel_array[0][254]
                    a5 = vel_array[0][255]
                    a6 = vel_array[0][0]
                    a7 = vel_array[1][254]
                    a8 = vel_array[1][255]
                    a9 = vel_array[1][0]

                    feature_dict[ii][jj] = [a1, a2, a3, a4, a5, a6, a7, a8, a9]

                elif ii == 255 and jj == 255:
                    # bottom right
                    total_count += 1
                    a1 = vel_array[254][254]
                    a2 = vel_array[254][255]
                    a3 = vel_array[254][0]
                    a4 = vel_array[255][254]
                    a5 = vel_array[255][255]
                    a6 = vel_array[255][0]
                    a7 = vel_array[0][254]
                    a8 = vel_array[0][255]
                    a9 = vel_array[0][0]

                    feature_dict[ii][jj] = [a1, a2, a3, a4, a5, a6, a7, a8, a9]

                elif ii == 0 and jj != 0 and jj != 255:
                    # top edge
                    total_count += 1
                    a1 = vel_array[255][jj-1]
                    a2 = vel_array[255][jj]
                    a3 = vel_array[255][jj+1]
                    a4 = vel_array[ii][jj-1]
                    a5 = vel_array[ii][jj]
                    a6 = vel_array[ii][jj+1]
                    a7 = vel_array[ii+1][jj-1]
                    a8 = vel_array[ii+1][jj]
                    a9 = vel_array[ii+1][jj+1]

                    feature_dict[ii][jj] = [a1, a2, a3, a4, a5, a6, a7, a8, a9]

                elif ii == 255 and jj != 0 and jj != 255:
                    # bottom edge
                    total_count += 1
                    a1 = vel_array[ii-1][jj-1]
                    a2 = vel_array[ii-1][jj]
                    a3 = vel_array[ii-1][jj+1]
                    a4 = vel_array[ii][jj-1]
                    a5 = vel_array[ii][jj]
                    a6 = vel_array[ii][jj+1]
                    a7 = vel_array[0][jj-1]
                    a8 = vel_array[0][jj]
                    a9 = vel_array[0][jj+1]

                    feature_dict[ii][jj] = [a1, a2, a3, a4, a5, a6, a7, a8, a9]

                elif jj == 0 and ii != 0 and ii != 255:
                    # left edge
                    total_count += 1
                    a1 = vel_array[ii-1][255]
                    a2 = vel_array[ii-1][jj]
                    a3 = vel_array[ii-1][jj+1]
                    a4 = vel_array[ii][255]
                    a5 = vel_array[ii][jj]
                    a6 = vel_array[ii][jj+1]
                    a7 = vel_array[ii+1][255]
                    a8 = vel_array[ii+1][jj]
                    a9 = vel_array[ii+1][jj+1]

                    feature_dict[ii][jj] = [a1, a2, a3, a4, a5, a6, a7, a8, a9]

                elif jj == 255 and ii != 0 and ii != 255:
                    # right edge
                    total_count += 1
                    a1 = vel_array[ii-1][jj-1]
                    a2 = vel_array[ii-1][jj]
                    a3 = vel_array[ii-1][0]
                    a4 = vel_array[ii][jj-1]
                    a5 = vel_array[ii][jj]
                    a6 = vel_array[ii][0]
                    a7 = vel_array[ii+1][jj-1]
                    a8 = vel_array[ii+1][jj]
                    a9 = vel_array[ii+1][0]

                    feature_dict[ii][jj] = [a1, a2, a3, a4, a5, a6, a7, a8, a9]
        # create_array(feature_dict)
        new_dict[direction] = create_array(feature_dict)
    return new_dict

def create_array(features):
    all_values = []
    for key_ii, values_ii in features.items():
        row_values = []
        for key_jj, values_jj in values_ii.items():
            row_values.append(values_jj)
        all_values.append(row_values)
    return np.array((all_values)).reshape(256*256, 9)

def my_reshaper(y):
    if isinstance(y, list):
        new_list = []
        for y_dict in y:
            new_dict = {}
            for key, values in y_dict.items():
                new_dict[key] = values.reshape(256*256, -1)
                new_list.append(new_dict)
        return new_list
    elif isinstance(y, dict):
        new_dict = {}
        for key, values in y.items():
            new_dict[key] = values.reshape(256*256, -1)
        return new_dict

    # print(new_array[0])
    # print(all_values[0][0] == new_array[0])
    # print(all_values[2][250] == new_array[256*2+250])
    # print('\n')


