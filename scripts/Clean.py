import csv
import json
import string
import re
def clean():
    #array to check against
    textarr = []
    #ticker and name arrays
    companyarr = ["Amazon","Apple","Facebook","Google","Netflix","Tesla"]
    tickerarr = ["AMZN", "AAPL", "FB", "GOOG", "NFLX", "TSLA", ]
    #for each company, pull all relavent tweets from files, strip of punctuation, and set to lowercase
    for i in range(0,6):
        f = open(companyarr[i]+".data", "w")
        a = open("WSJ.csv", encoding="utf8")
        b = open("Bloomberg.csv", encoding="utf8")
        c = open("FT.csv", encoding="utf8")
        csv_a = csv.reader(a)
        csv_b = csv.reader(b)
        csv_c = csv.reader(c)
        for row in csv_a:
            if (((companyarr[i] or tickerarr[i]) in row[10]) and row is not None):
                text = row[10][0:30]
                if text not in textarr:
                    textarr.append(text)
                    text2 = row[10]
                    text2 = text2.replace("\u2019", "'")
                    text2 = text2.replace("\u2014", "-")
                    text2 = re.sub(r'[^\w\s]','',text2)
                    text2 = text2.lower()
                    dict = {"date":row[3],"time":row[4],"text":text2}
                    f.write(json.dumps(dict)+"\n")
        for row in csv_b:
            if (((companyarr[i] or tickerarr[i]) in row[10]) and row is not None):
                text = row[10][0:30]
                if text not in textarr:
                    textarr.append(text)
                    text2 = row[10]
                    text2 = text2.replace("\u2019", "'")
                    text2 = text2.replace("\u2014", "-")
                    text2 = re.sub(r'[^\w\s]','',text2)
                    text2 = text2.lower()
                    dict = {"date": row[3], "time": row[4], "text": text2}
                    f.write(json.dumps(dict) + "\n")
        for row in csv_c:
            if (((companyarr[i] or tickerarr[i]) in row[10]) and row is not None):
                text = row[10][0:30]
                if text not in textarr:
                    textarr.append(text)
                    text2 = row[10]
                    text2 = text2.replace("\u2019", "'")
                    text2 = text2.replace("\u2014", "-")
                    text2 = re.sub(r'[^\w\s]','',text2)
                    text2 = text2.lower()
                    dict = {"date": row[3], "time": row[4], "text": text2}
                    f.write(json.dumps(dict) + "\n")
        f.close()
        a.close()
        b.close()
        c.close()

if __name__ == '__main__':
    clean()
