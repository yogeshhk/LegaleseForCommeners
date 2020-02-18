"""
Robot Lawyer Demo App : Finding Judgement Similarity

Author: Yogesh H Kulkarni
Last modified: 18 November 2016
"""
from tkinter import *

class robotUI(Frame):
    def __init__(self, parent, rDf):
        Frame.__init__(self, parent)
        self.robotDataFrame = rDf
        self.parent = parent

        self.keywordsAppText = "Keywords: <keyword 1, keyword 2,...>"
        self.summaryAppText = "<Summary will appear here ...>"
        self.searchAppText = "<Enter word here ...>"
        self.judgementAppText = "<Full Judgement will appear here ...>"
        self.similarityAppList = ["...", "...", "...", "..."]
        self.searchAppList = ["...", "...", "...", "..."]
        self.titleAppText = "Legal Analytics"
        self.authorAppText = "Send comments to yogeshkulkarni@yahoo.com"

        self.initUI()

    def cbSearchButtonClick(self):
        searchWord = self.searchEntry.get()
        print("Word entered {}".format(searchWord))
        self.searchAppList = self.robotDataFrame.querySearchedJudgementFilenamesBySearchItem(searchWord)
        self.searchList.delete(0,END)
        for i, v in enumerate(self.searchAppList):
            if i < 6: # Limit to 5 judgements for now
                self.searchList.insert(i + 1, v)

    def cbOnSelectSearchList(self,event):
        w = event.widget
        index = int(w.curselection()[0])
        value = w.get(index)
        print("You selected item in Search List {}: {}".format(index, value))

        self.similarityAppList = self.robotDataFrame.querySimilarJudgementFilenamesByGivenJudgement(value)
        self.similarityList.delete(0,END)
        for i, v in enumerate(self.similarityAppList):
            if i < 6: # Limit to 5 judgements for now
                self.similarityList.insert(i + 1, v)

        self.keywordsAppText = self.robotDataFrame.queryKeywordsByFilename(value)
        self.keywordsLabel['text'] = "Keywords : " + self.keywordsAppText

        self.judgementAppText = self.robotDataFrame.queryFullJudgementByFilename(value)
        self.fullText.delete(1.0,END)
        self.fullText.insert(1.0, self.judgementAppText)

        self.summaryAppText = self.robotDataFrame.querySummaryByFilename(value)
        self.summaryText.delete(1.0,END)
        self.summaryText.insert(1.0, self.summaryAppText)

    def cbOnSelectSimilarityList(self,event):
        w = event.widget
        index = int(w.curselection()[0])
        value = w.get(index)
        print("You selected item  in Similarity List {}: {}".format(index, value))

        self.keywordsAppText = self.robotDataFrame.queryKeywordsByFilename(value)
        self.keywordsLabel['text'] = "Keywords : " + self.keywordsAppText

        self.judgementAppText = self.robotDataFrame.queryFullJudgementByFilename(value)
        self.fullText.delete(1.0,END)
        self.fullText.insert(1.0, self.judgementAppText)

        self.summaryAppText = self.robotDataFrame.querySummaryByFilename(value)
        self.summaryText.delete(1.0,END)
        self.summaryText.insert(1.0, self.summaryAppText)


    def initUI(self):
        self.parent.title(self.titleAppText)
        self.pack()

        topFrame = Frame(self)
        topFrame.grid(row=0)

        self.title = Label(topFrame, text=self.titleAppText, font="Helvetica 20 bold")
        self.title.pack(side=LEFT)

        # ----------- Left side ---------------

        leftFrame = Frame(self)
        leftFrame.grid(row=1, column=0)

        # ------------
        searchBoxFrame = Frame(leftFrame)
        searchBoxFrame.pack(fill=BOTH, expand=True)

        # searchLabel = Label(searchBoxFrame, text="Enter Keyword")
        # searchLabel.pack(fill=BOTH, expand=True)

        self.searchEntry = Entry(searchBoxFrame, text=self.searchAppText)
        self.searchEntry.pack(side=LEFT)

        searchButton = Button(searchBoxFrame, text="Search", command=self.cbSearchButtonClick)
        searchButton.pack(side=LEFT)

        # --------------
        searchFilesFrame = Frame(leftFrame)
        searchFilesFrame.pack(fill=BOTH, expand=True)

        searchListTitle = Label(searchFilesFrame, text="Searched Judgements")
        searchListTitle.pack(fill=BOTH, pady=5, padx=5, expand=True)

        self.searchList = Listbox(searchFilesFrame, selectmode=SINGLE)
        for i, v in enumerate(self.searchAppList):
            if i < 6: # add only 5 for now
                self.searchList.insert(i + 1, v)
        self.searchList.bind('<<ListboxSelect>>', self.cbOnSelectSearchList)
        self.searchList.pack(fill=BOTH, expand=True)

        # --------------
        similarFilesFrame = Frame(leftFrame)
        similarFilesFrame.pack(fill=BOTH, expand=True)

        similarityListTitle = Label(similarFilesFrame, text="Similar Judgements")
        similarityListTitle.pack(fill=BOTH, pady=5, padx=5, expand=True)

        self.similarityList = Listbox(similarFilesFrame, selectmode=SINGLE)
        for i, v in enumerate(self.similarityAppList):
            if i < 6: # add only 5 for now
                self.similarityList.insert(i + 1, v)
        self.similarityList.bind('<<ListboxSelect>>', self.cbOnSelectSimilarityList)
        self.similarityList.pack(fill=BOTH, expand=True)

        # -------------
        keywordsFrame = Frame(leftFrame)
        keywordsFrame.pack(fill=BOTH, expand=True)

        self.keywordsLabel = Label(keywordsFrame, text=self.keywordsAppText, height=2)
        self.keywordsLabel.pack(side=LEFT)

        # ----------- Right side -------------------

        rightFrame = Frame(self)
        rightFrame.grid(row=1, column=1)

        # ------------
        fullTextFrame = Frame(rightFrame)
        fullTextFrame.pack(fill=BOTH, expand=True)

        fullTextTitle = Label(fullTextFrame, text="Judgement")
        fullTextTitle.pack(fill=BOTH, pady=5, padx=5, expand=True)

        self.fullText = Text(fullTextFrame, borderwidth=3, relief="sunken")
        self.fullText.config(undo=True, wrap='word')
        self.fullText.insert(INSERT, self.judgementAppText)
        self.fullText.pack(fill=BOTH, pady=5, padx=5, expand=True)
        # scrollb = Scrollbar(self.fullText, command=self.fullText.yview)
        # scrollb.pack(side=LEFT, pady=5, padx=5, expand=True)
        # self.fullText['yscrollcommand'] = scrollb.set
        # ------------
        summaryFrame = Frame(rightFrame)
        summaryFrame.pack(fill=BOTH, expand=True)

        summaryTextTitle = Label(summaryFrame, text="Summary")
        summaryTextTitle.pack(fill=BOTH, pady=5, padx=5, expand=True)

        self.summaryText = Text(summaryFrame, height=3, borderwidth=3, relief="sunken")
        self.summaryText.config(undo=True, wrap='word')
        self.summaryText.insert(INSERT, self.summaryAppText)
        self.summaryText.pack(fill=BOTH, pady=5, padx=5, expand=True)

        # ----------- Bottom Side --------------------

        bottomFrame = Frame(self)
        bottomFrame.grid(row=2)

        # ----------------

        bootomLabel = Label(bottomFrame, text=self.authorAppText, height=1)
        bootomLabel.pack(side=LEFT)

def main():
    root = Tk()
    app = robotUI(root,None)

    root.mainloop()

if __name__ == '__main__':
    main()