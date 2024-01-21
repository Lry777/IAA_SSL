import os
import torch
import numpy as np
from torchvision import transforms
import json
from sklearn.model_selection import train_test_split
import glob
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from torchvision.datasets.folder import default_loader
from pathlib import Path
from tqdm import tqdm
import os
import pandas as pd

logger = logging.getLogger(__file__)

def _remove_all_not_found_image(df: pd.DataFrame, path_to_images: Path) -> pd.DataFrame:
    clean_rows = []
    for _, row in df.iterrows():
        image_id = row["image_id"]
        try:
            # file_name = path_to_images / f"{image_id}.jpg"
            file_name = os.path.join(path_to_images, str(image_id)+'.jpg')
            # print(file_name)
            _ = default_loader(file_name)
        except (FileNotFoundError, OSError, UnboundLocalError) as ex:
            logger.info(f"broken image {file_name} : {ex}")
        else:
            clean_rows.append(row)
    df_clean = pd.DataFrame(clean_rows)
    return df_clean

def remove_all_not_found_image(df: pd.DataFrame, path_to_images: Path, num_workers: int) -> pd.DataFrame:
    futures = []
    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for df_batch in np.array_split(df, num_workers):
            future = executor.submit(_remove_all_not_found_image, df=df_batch, path_to_images=path_to_images)
            futures.append(future)
        for future in tqdm(as_completed(futures), total=len(futures)):
            results.append(future.result())
    new_df = pd.concat(results)
    return new_df

def read_ava_txt(path_to_ava: Path) -> pd.DataFrame:
    # NIMA origin file format and indexes
    df = pd.read_csv(path_to_ava, header=None, sep=" ")
    del df[0]
    score_first_column = 2
    score_last_column = 12
    tag_first_column = 1
    tag_last_column = 4
    score_names = [f"score{i}" for i in range(score_first_column, score_last_column)]
    tag_names = [f"tag{i}" for i in range(tag_first_column, tag_last_column)]
    df.columns = ["image_id"] + score_names + tag_names
    # leave only score columns
    df = df[["image_id"] + score_names]
    return df

def get_clean_data(data_path):
    if os.path.exists(os.path.join(data_path, 'clean_all.csv')):
        return pd.read_csv(os.path.join(data_path, 'clean_all.csv'))
    else:
        df = read_ava_txt(os.path.join(data_path, 'AVA.txt'))
        targets = remove_all_not_found_image(df, os.path.join(data_path, 'images/images'), num_workers=4)
        targets.to_csv(os.path.join(data_path, 'clean_all.csv'), index=False)
        return targets

def get_train_test(train_test_path, target):
    train_path = glob.glob(os.path.join(train_test_path,'*_train.jpgl'))
    test_path = glob.glob(os.path.join(train_test_path,'*_test.jpgl'))

    columns = target.columns
    target = target.values.tolist()
    train_rows = []
    test_rows = []
    # for tr_p in train_path:
    #     with open(tr_p, 'r') as f:
    #         for idx in f:
    #             print(idx)
    #             row = find_idx_rows(target, idx)
    #             if row != []:
    #                 train_rows.append(row)

    for ts_p in test_path:
        print(test_path)
        with open(ts_p, 'r') as f:
            for idx in f:
                print(idx)
                row = find_idx_rows(target, idx)
                if row != []:
                    test_rows.append(row)
                    target.remove(row)
    # df_train = pd.DataFrame(train_rows)
    # print(len(target), len(test_rows))
    df_test = pd.DataFrame(test_rows)
    df_test.columns = columns
    target = pd.DataFrame(target)
    target.columns = columns
    return target, df_test



def find_idx_rows(all_list, idx):
    # print(all_list[248988])
    for i, row in enumerate(all_list):
        # print(row[0])
        if int(row[0]) == int(idx):
            # print('yep',i)
            return row
        elif i == len(all_list)-1:
            print('no idx:', idx)
            return []

def get_cl_data(path):
    cl_train_path = os.path.join(path, 'train.jpgl')
    cl_lable_path = os.path.join(path, 'train.lab')
    cl_train = []
    with open(cl_train_path, 'r') as f:
        for idx in f:
            cl_train.append([int(idx)])

    with open(cl_lable_path, 'r') as f:
        for i, cl in enumerate(f):
            id_image = cl_train[i]
            id_image.append(int(cl))
    # print(cl_train)
    cl_train = pd.DataFrame(cl_train)
    columns = ['image_id','cl']
    cl_train.columns = columns

    return cl_train

def get_cl_all_csv(df_cl, target):
    columns = list(target.columns)
    # print(columns)
    # columns.append('cl')
    target = target.values.tolist()
    df_cl_li = df_cl.values.tolist()
    ft_data = []
    for cl_li in df_cl_li:
        image_id = cl_li[0]
        # image_cl = cl_li[1]
        print('image_id:', image_id)
        for i, row in enumerate(target):
            if row[0] == image_id:
                # row.append(image_cl)
                ft_data.append(row)
                break
            elif i == len(target) - 1:
                print('no idx:', image_id)

    df_ft_data = pd.DataFrame(ft_data)
    df_ft_data.columns = columns
    return df_ft_data

def add_cl_to_lbdata(df_lb, df_cl):
    columns = list(df_lb.columns)
    # print(columns)
    columns.append('cl')

    li_lb = df_lb.values.tolist()
    li_cl = df_cl.values.tolist()

    for row in li_lb:
        image_id = row[0]
        for i, r in enumerate(li_cl):
            if image_id == r[0]:
                row.append(r[1])
                break
            elif i == len(li_cl)-1:
                row.append(-1)
    df_lb = pd.DataFrame(li_lb)
    df_lb.columns = columns
    return df_lb

def mix_cl_and_lb(df_cl, df_lb):
    columns = list(df_lb.columns)
    li_lb = df_lb.values.tolist()
    li_cl = df_cl.values.tolist()

    mix = li_lb[::]
    for i, cl in enumerate(li_cl):
        image_id = cl[0]
        b = 1
        for lb in li_lb:
            if lb[0] == image_id:
                b = 0
                pass

        if b==1:
            mix.append(cl)

    print(len(mix))
    mix = pd.DataFrame(mix)
    mix.columns = columns
    return mix

def avg_split_traindata(target, path, rate=0.1):
    columns = list(target.columns)
    target = target.values.tolist()
    print('target len:', len(target))
    train_path = glob.glob(os.path.join( path, '*_train.jpgl'))

    sp_train=[]
    for path in train_path:
        df = pd.read_csv(path)
        df_lb, df_ulb = train_test_split(df, train_size=rate)

        li_lb = df_lb.values.tolist()
        print('part len:', len(li_lb))
        for i in li_lb:
            idx = i[0]
            # print(idx)
            row = find_idx_rows(target, idx)
            if row != []:
                sp_train.append(row)
                target.remove(row)
    print('total len:',len(sp_train))
    df_sp_train = pd.DataFrame(sp_train)
    df_sp_train.columns = columns
    return df_sp_train

if __name__ == "__main__":
    data_path = '/home/xiexie/data/AVA_dataset'
    target = get_clean_data(data_path)
    # # 整理出测试集和训练集
    # train_test_path = os.path.join(data_path, 'aesthetics_image_lists')
    # df_train, df_test = get_train_test(train_test_path, target)
    # df_train.to_csv(os.path.join(data_path,'train.csv'), index=False)
    # df_test.to_csv(os.path.join(data_path,'test.csv'), index=False)
    # 从训练集中划分有标签和无标签
    # df_train = pd.read_csv(os.path.join(data_path,'train.csv'))
    # df_lb, df_ulb = train_test_split(df_train, train_size=0.2)
    # df_lb.to_csv(os.path.join(data_path, 'lb_0.2.csv'), index=False)
    # df_lb = avg_split_traindata(target=df_train,path=train_test_path,rate=0.1)


    # 划分 分类数据
    cl_data_path = os.path.join(data_path, 'style_image_lists')
    # df_cl = get_cl_data(cl_data_path)
    # # df_cl.to_csv(os.path.join(cl_data_path, 'cl.csv'), index=False)
    #
    # df_ft_data = get_cl_all_csv(df_cl, target)
    # df_ft_data.to_csv(os.path.join(cl_data_path, 'avg_data.csv'), index=False)

    # 对有标签数据添加分类信息
    # df_lb = pd.read_csv(os.path.join(data_path, 'lb_0.1.csv'))
    # df_cl = pd.read_csv(os.path.join(cl_data_path, 'cl.csv'))
    # df_lb = add_cl_to_lbdata(df_lb, df_cl)
    # df_lb.to_csv(os.path.join(data_path, 'lb_0.1_cl.csv'), index=False)

    # 将分类数据与随机有标签数据融合
    # df_cl_avg = pd.read_csv(os.path.join(cl_data_path, 'avg_data.csv'))
    # df_lb = pd.read_csv(os.path.join(data_path,'lb_0.01.csv'))
    # mix = mix_cl_and_lb(df_cl_avg, df_lb)
    # mix.to_csv(os.path.join(data_path, 'mix_lb_cl_0.01.csv'), index=False)

    print(len(pd.read_csv(os.path.join(cl_data_path, 'avg_data.csv'))))
    print(len(pd.read_csv(os.path.join(data_path,'lb_0.01.csv'))))
    pp = os.path.join(data_path, 'mix_lb_cl_0.01.csv')
    df_lb = pd.read_csv(pp)
    print(len(df_lb))








