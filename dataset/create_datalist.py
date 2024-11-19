import os



def make_datalist(data_fd, data_list):

    filename_all = os.listdir(data_fd)

    filename_all = sorted([data_fd + '/' + img_name + '\n' for img_name in filename_all if img_name.endswith('.npy')],key=lambda name: int(name[52:-5]))

    with open(data_list, 'w') as fp:
        fp.writelines(filename_all)




if __name__ == '__main__':

    #Plz change the path follow your setting
    data_fd      = '/home/zhr/MPSCL/data/data_np/train_mr'
    data_list    = '../data/datalist/train_mr.txt'
    make_datalist(data_fd, data_list)

