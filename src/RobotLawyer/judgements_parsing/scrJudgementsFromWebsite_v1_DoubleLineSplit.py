import urllib2
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

sc_site_url = 'http://judis.nic.in/supremecourt/imgst.aspx?'

class Judgement:
    def __init__(self, file_id, sc_judgement_url,JudgementTitle,AppellantName,RespondantName, judges,paragraphs_list):
        self.file_id = file_id
        self.sc_judgement_url = sc_judgement_url
        self.JudgementTitle = JudgementTitle
        self.AppellantName = AppellantName
        self.RespondantName = RespondantName
        self.judges = list(set(judges))
        self.paragraphs_list = paragraphs_list

    def __str__(self):
        id =  "------------ File Id : " + str(self.file_id) + "------------------\n"
        link =  "Link : " + self.sc_judgement_url + "\n"
        title =  "Title : " + self.JudgementTitle + "\n"
        applnt =  "AppellantName : " + self.AppellantName + "\n"
        resp =  "RespondantName : " + self.RespondantName + "\n"
        judge = "Judge(s) : " + " | ".join(self.judges) + "\n"
        #sample_para = self.paragraphs_list[0]  + "\n"
        return id + link + title + applnt + resp + judge # + sample_para

import re
def process_string(text):
    lines = text.splitlines()
    line = ' '.join(lines)
    clnline = re.sub('\s+', ' ', line).strip()
    return clnline

def read_judgement_from_website(id):
    case_number = 'filename=' + str(id) #43216'
    sc_judgement_url = sc_site_url + case_number

    #Query the website and return the html to the variable 'page'
    page = urllib2.urlopen(sc_judgement_url)

    #import the Beautiful soup functions to parse the data returned from the website
    from bs4 import BeautifulSoup

    #Parse the html in the 'page' variable, and store it in Beautiful Soup format
    soup = BeautifulSoup(page, "html.parser")

    JudgementTitle = ""
    AppellantName = ""
    RespondantName = ""
    judges_list = []
    content_list = []

    try:
        content = unicode(soup.textarea.string)  # no paragraphs, no links
        paragraphs = content.split("\n\n")

        for para in paragraphs:

            if len(para) < 1:
                continue

            if para.find(u"REPORTABLE") != -1 or para.find(u"VERSUS") != -1 or para.find(
                            u"J U D G M E N T") != -1 or para.find(u"IN THE SUPREME COURT OF INDIA") != -1:
                continue

            if para.find(u"CIVIL APPEAL") != -1 or para.find(u"WRIT PETITION") != -1 or para.find(u"CRIMINAL APPEAL") != -1:
                JudgementTitle = process_string(para)
                continue

            if para.find(u"APPELLANT") != -1 or para.find(u"PETITIONER") != -1:
                AppellantName = process_string(re.split(r'\s{2,}', para)[0])
                continue

            if para.find(u"RESPONDENT") != -1:
                RespondantName = process_string(re.split(r'\s{2,}', para)[0])
                continue

            if para.find(u".J.") != -1  or para.find(u"..J.") != -1 or para.find(u"J.\n") != -1 or para.find(u"J. ") != -1:
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

    return Judgement(id,sc_judgement_url,JudgementTitle,AppellantName,RespondantName, judges_list, content_list)

judgement_objs = []
query_judgements = [43215,43216, 43218, 43123]
for i in query_judgements:
    judgement = read_judgement_from_website(i)
    print judgement
    judgement_objs.append(judgement)
