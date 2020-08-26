from bs4 import BeautifulSoup
import urllib.request as req
#入力した銘柄の決算短信が発表された日付をとってくる。
name = input()
url = "http://ke.kabupro.jp/code/"+ name +".htm"

i = 0
#file = open( name + '_tanshin.csv', 'a')

res = req.urlopen(url)

soup = BeautifulSoup(res, "html.parser")

data = soup.select("table.Quote > tr > td.CellKessan")
print("==以下は決算短信日==")
for th in data:
    sd = th.string
    sd = sd.split('/')
    sd = str(sd[0]+"年"+str(int(sd[1]))+"月"+str(int(sd[2]))+"日")
    print(sd)
    #file.writelines(sd + "\n")

url2 = "http://www.kabupro.jp/yuho/"+ name + ".htm"
res = req.urlopen(url2)
soup = BeautifulSoup(res, "html.parser")

data = soup.select("table.Quote > tr > td.CellKessan")
print("==以下は有価証券情報開示日==")
for th in data:
    sd = th.string
    sd = sd.split('/')
    sd = str(sd[0]+"年"+str(int(sd[1]))+"月"+str(int(sd[2]))+"日")
    print(sd)
#file.close()
