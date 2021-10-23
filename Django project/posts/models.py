from django.db import models
from django.shortcuts import redirect, render

# Create your models here.
def searchdata(movieurl):
    from bs4 import BeautifulSoup
    import requests
    import re
    # data = requests.get('https://www.rottentomatoes.com/top/bestofrt/top_100_action__adventure_movies/')
    # https://www.rottentomatoes.com/top/bestofrt/top_100_action__adventure_movies/
    data = requests.get(movieurl,headers={'Accept-Language':'en'})
    soup = BeautifulSoup(data.text,'html.parser')

    # print(soup.get_text())
    list = []
    a = soup.find_all('a',{"class":"unstyled articleLink"})
    # print(a)
    for i in a:
    #     # x = i.find('a')
        x = str(i.text).replace('\n','').replace(' ','')
        # print(x)
        dic = [' \n','X']
        if x in dic:
            pass
        else:
            list.append(x)
    return list

# data = requests.get('https://www.rottentomatoes.com/top/bestofrt/top_100_action__adventure_movies/',headers={'Accept-Language':'en'})
# soup = BeautifulSoup(data.text,'html.parser')

# # print(soup.get_text())unstyled articleLink
# list = []
# a = soup.find_all('a', {"class":"unstyled articleLink"})
# # print(a)
# for i in a:
# #     # x = i.find('a')
#     x = str(i.text).replace('\n','').replace(' ','')
#     print(x)
#     dic = [' \n','X']
#     if x in dic:
#         pass
#     else:
#         list.append(x)