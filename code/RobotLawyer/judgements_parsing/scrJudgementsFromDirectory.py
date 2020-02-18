import urllib2
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import os
path = '../Datasets/nd_judgements/test'

class Judgement:
    def __init__(self, JudgementTitle,AppellantName,RespondantName, judges,paragraphs_list):
        self.JudgementTitle = JudgementTitle
        self.AppellantName = AppellantName
        self.RespondantName = RespondantName
        self.judges = list(set(judges))
        self.paragraphs_list = paragraphs_list

    def __str__(self):
        title =  "Title : " + self.JudgementTitle + "\n"
        applnt =  "AppellantName : " + self.AppellantName + "\n"
        resp =  "RespondantName : " + self.RespondantName + "\n"
        judge = "Judge(s) : " + " | ".join(self.judges) + "\n"
        #sample_para = self.paragraphs_list[0]  + "\n"
        return title + applnt + resp + judge # + sample_para

import re
def process_string(text):
    lines = text.splitlines()
    line = ' '.join(lines)
    clnline = re.sub('\s+', ' ', line).strip()
    return clnline

def parse_judgement_text(content):
    JudgementTitle = ""
    AppellantName = ""
    RespondantName = ""
    judges_list = []
    content_list = []

    try:

        paragraphs = content.split("\n\n")

        for para in paragraphs:

            if len(para) < 1:
                continue

            if para.find(u"REPORTABLE") != -1 or para.find(u"VERSUS") != -1 or para.find(
                    u"J U D G M E N T") != -1 or para.find(u"IN THE SUPREME COURT OF INDIA") != -1:
                continue

            if para.find(u"CIVIL APPEAL") != -1 or para.find(u"WRIT PETITION") != -1 or para.find(
                    u"CRIMINAL APPEAL") != -1:
                JudgementTitle = process_string(para)
                continue

            if para.find(u"APPELLANT") != -1 or para.find(u"PETITIONER") != -1:
                AppellantName = process_string(re.split(r'\s{2,}', para)[0])
                continue

            if para.find(u"RESPONDENT") != -1:
                RespondantName = process_string(re.split(r'\s{2,}', para)[0])
                continue

            if para.find(u".J.") != -1 or para.find(u"..J.") != -1 or para.find(u"J.\n") != -1 or para.find(
                    u"J. ") != -1:
                JudgesName = ""
                if para.find("[") != 1 and para.find("]") != -1:
                    JudgesName = process_string(para[para.find("[") + 1:para.find("]")])
                if para.find("(") != 1 and para.find(")") != -1:
                    JudgesName = process_string(para[para.find("(") + 1:para.find(")")])
                if len(JudgesName) > 1:
                    judges_list.append(JudgesName)
                continue

            content_list.append(para)

    except AttributeError:
        print "Oops!  That was no valid.  Try again..."

    return Judgement(JudgementTitle, AppellantName, RespondantName, judges_list, content_list)


def read_judgement_from_directory(filename):
    fullreadfilename = path + "/read/" + filename
    num_skip = 2 # first two lines were junk, 3rd onwards was html dump
    with open(fullreadfilename) as rf:
        page_in_list = rf.readlines()[num_skip:]

    page = " ".join(page_in_list)
    from bs4 import BeautifulSoup
    #Parse the html in the 'page' variable, and store it in Beautiful Soup format
    soup = BeautifulSoup(page, "html.parser")
    textareas = soup.findAll('textarea')
    content = unicode(textareas[0].text)  # no paragraphs, no links
    if content:
        fullwritefilename = path + "/write/" + filename
        wf = open(fullwritefilename, 'w')
        wf.write(content.encode('utf8'))
        wf.close()
    return content



judgement_objs = []
for filename in os.listdir(path + "/read"):
    content = read_judgement_from_directory(filename)
    judgement = parse_judgement_text(content)
    print judgement
    judgement_objs.append(judgement)
