from bs4 import BeautifulSoup as bs
from urllib.request import urlopen


def parse(url):
    page = urlopen(url)
    soup = bs(page, "html.parser")
    return soup

def get_data_from_rakuten(soup):

    
    title = soup.find("span",attrs={'class': "detailHeadline"}).string
    image_url = soup.find("a",attrs={'class': "prdMainPhoto"}).img["src"]
    informations = soup.find("div",attrs={'id': "prd_information"})

    texts = " ".join([info.text for info in informations])

    return {
        "text_input" : title ,#+ " " + texts,
        "url_input" : image_url,
        "provider" : "Rakuten"
    }


def get_data_from_rueducommerce(soup):

    
    title = soup.find("h1",attrs={'class': "product-name"}).findAll("span")[1].text
    try:
        legend = soup.find("p",attrs={'class': "legende"}).text
    except:
        legend = ""
    informations = soup.find("div",attrs={'class': "content"})

    texts = " ".join([info.text for info in informations])

    return {
        "text_input" : title + " " + legend ,#+ " " + texts,
        "url_input" : "",
        "provider" : "RueDuCommerce"
    }

def scrap(url):
    soup = parse(url)

    if "rueducommerce" in url:
        print("scrapp get_data_from_rueducommerce")
        return get_data_from_rueducommerce(soup)
    elif "rakuten" in url:
        print("scrapp rakuten")
        return get_data_from_rakuten(soup)

if __name__ == "__main__":

    url = "https://www.rueducommerce.fr/p-samsung-galaxy-z-fold4-12256-go-5g-noir-smartphone-pliable-samsung-3407624-27453.html"
    
    response = scrap(url)
    print(response)