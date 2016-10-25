import requests
import json
import pprint
import lxml
import os

uuu="http://www.shihuo.cn/shihuo/getList?pageSize=20&createTimestamp="
get_next="//li[@class='clearfix']/@data-time[last()]"
url='http://www.shihuo.cn/'

x=html.parse(url)
next_dex=x.xpath(get_next)
url_new=uuu+next_dex
cont=requests.get(url_new).content
con=cont.decode('unicode_escape')
fin=json.loads(con)
pprint.pprint(fin['data'][0]['data'])
