import matplotlib.pyplot as plt
import pandas as pd
from os import listdir
from os.path import isfile, join
import numpy as np
import seaborn as sns

from pit import pit_versions


def evaluate_train_loss():
    # df = pd.read_csv('loss2.csv',
    #                names=['train_overall_loss', 'train_value_policy', 'train_policy_loss'])
    # s = df.loc[:, 'train_overall_loss']
    # last_occurence = s.where(s == 1.56760).last_valid_index()
    # newDf = df.iloc[last_occurence::, :]
    # newDf.to_csv('loss2.csv', index=False)
    # newDf.reset_index()

    df = pd.read_csv('loss.csv', names=['policy_loss', 'value_loss'])
    df.plot.line()
    plt.xlabel('Iteration number')
    plt.ylabel('Loss')
    plt.show()


def get_models_from_path(path):
    models = [f for f in listdir(path) if isfile(join(path, f))]
    versions_nums = []
    for name in models:
        if "00001" in name and "checkpoint" in name:
            # split_name = name.split(".")[0:3]
            # version_name = '.'.join(split_name)
            checkpoint_name = name.split(".")[0]
            checkpoint_num = int(checkpoint_name.split("_")[1])
            versions_nums.append(checkpoint_num)

    versions_nums.sort()
    return versions_nums


def run_tournament():
    path_to_models = "C://Users//edoma//PycharmProjects//alpha-zero-general-forked//server_models//run1googleInstance//versions"
    # versions_nums = get_models_from_path(path_to_models)
    versions_nums = [2, 10, 20, 33, 50]

    df = pd.DataFrame(data=np.zeros([len(versions_nums), len(versions_nums)], dtype=np.int8), index=versions_nums,
                      columns=versions_nums)
    print(df)

    for rowIndex, row in df.iterrows():  # iterate over rows
        for columnIndex, value in row.items():
            if rowIndex != columnIndex:
                if rowIndex > columnIndex:
                    continue
                version1_name = 'checkpoint_{}.pth.tar'.format(rowIndex)
                version2_name = 'checkpoint_{}.pth.tar'.format(columnIndex)
                first_won, second_won, draws = pit_versions(version1_name, version2_name)
                df.loc[rowIndex, columnIndex] = first_won - second_won
                df.loc[columnIndex, rowIndex] = second_won - first_won

    df.to_csv('tournament_result.csv', index=False)


def evaluate_tournament():
    df = pd.read_csv('tournament_result.csv')

    df['version_number'] = df.columns
    df['points'] = df.sum(axis=1)

    sns.set()
    sns.lineplot(x='version_number', y='points', legend=False, markers=["o"], style=True, data=df)
    plt.xlabel("NN version number")
    plt.ylabel("Points")
    plt.legend(title='Max points gain: 8')
    plt.show()


evaluate_train_loss()
# run_tournament()
# evaluate_tournament()
