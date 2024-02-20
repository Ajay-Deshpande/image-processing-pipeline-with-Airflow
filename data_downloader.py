import bs4
import requests
from urllib import request
from urllib.parse import quote
import os
from tqdm import tqdm


def write_img_file(img, category, file_num):
    folder_name = f'images/{category}'
    file_name = f'{folder_name}/image_{file_num}.jpg'
    try:
        with open(file_name, 'wb') as f:
            f.write(img)
    except:
        os.makedirs(folder_name)
        with open(file_name, 'wb') as f:
            f.write(img)
    return True

def main():
    HEADERS = {
        'User-Agent':
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.90 Safari/537.36"
    }

    image_classes = ["bicycle", 'motorcycle', 'car', 'truck', 'aeroplane', 'helicopter', 'ship']

    for keyword in image_classes:
        resp = request.Request(f"https://www.google.com/search?q={quote(keyword)}&tbm=isch", headers = HEADERS)
        resp = str(request.urlopen(resp).read())
        soup = bs4.BeautifulSoup(resp)
        urls_ = soup.find_all('img')
        urls_ = [i.attrs['data-src'] for i in urls_ if i.attrs.get('data-src') and i.attrs['data-src'].find('images?') != -1]
        counter = tqdm(urls_)
        for index, url in enumerate(urls_):        
            try:
                img = requests.get(url).content
            except:
                continue
            counter.update()
            write_img_file(img, keyword, index)

if __name__ == "__main__":
    main()