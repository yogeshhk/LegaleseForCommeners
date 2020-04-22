"""
Robot Lawyer Demo App : Finding Judgement Similarity

Author: Yogesh H Kulkarni
Last modified: 18 November 2016
"""
from tkinter import *
from ui import robotUI
from dataframe import robotDataFrame


def main():
    path = '../Datasets/yhk_cleaned_judgements/test/'
    rDf = robotDataFrame(path)

    root = Tk()
    app = robotUI(root,rDf)

    root.mainloop()

if __name__ == '__main__':
    main()