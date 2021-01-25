import requests
from bs4 import BeautifulSoup as bs

final = ""

def get_video_info(url):
    content = requests.get(url)
    soup = bs(content.content, "html.parser")
    result = {}
    result['views'] = int(soup.find("div", attrs={"class": "watch-view-count"}).text[:-6].replace(",", ""))
    result['likes'] = int(soup.find("button", attrs={"title": "I like this"}).text.replace(",", ""))
    result['dislikes'] = int(soup.find("button", attrs={"title": "I dislike this"}).text.replace(",", ""))
    return result


f = open("D:\\MiniProject\\Final\\YT-Records.txt","r")
all_info = f.readlines()
for info in all_info:
	entry = info.replace('\n','')
	entry_list = entry.split(',')
	t_id = entry_list[0]
	url = entry_list[1]
	laste = int(entry_list[2])
	if laste != (len(entry_list) - 1):
		nexte = laste + 1
		result = get_video_info(url)
		entry_list[nexte] = str(result['views'])
		entry_list[nexte+1] = str(result['likes'])
		entry_list[nexte+2] = str(result['dislikes'])
		entry_list[2] = nexte + 2
	for e in entry_list:
		final += str(e) + ','
	final = final[:-1]
	final += '\n'
f.close()

f = open("D:\\MiniProject\\Final\\YT-Records.txt","w")
f.write(final)
f.close()