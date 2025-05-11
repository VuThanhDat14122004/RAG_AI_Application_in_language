from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import GPT4AllEmbeddings
# import beautifulsoup4
import re
import requests
from bs4 import BeautifulSoup
import json


headers = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
}

data_folder_dir = "data"

website_url_info_history_pb = ["https://en.wikipedia.org/wiki/Pittsburgh", "https://en.wikipedia.org/wiki/History_of_Pittsburgh",
                    "https://www.britannica.com/place/Pittsburgh", "https://www.visitpittsburgh.com/",
                    "https://www.pittsburghpa.gov/City-Government/Finances-Budget/Taxes/Tax-Forms"]
pdf_url_info_history_pb = ["https://apps.pittsburghpa.gov/redtail/images/23255_2024_Operating_Budget.pdf"]

special_website_url_event_pb = ["https://downtownpittsburgh.com/events/"]
# lấy bằng hàm special

website_url_event_pb = ["https://www.pghcitypaper.com/pittsburgh/EventSearch?v=d"]

website_url_music_pb = ["https://pittsburghopera.org/"]
website_url_museums_pb = ["https://carnegiemuseums.org/", "https://www.heinzhistorycenter.org/",
                          "https://www.thefrickpittsburgh.org/", "https://en.wikipedia.org/wiki/List_of_museums_in_Pittsburgh"]
# link cuối trong list trên có table, cần lấy bằng hàm có table

website_url_food_pb = ["https://www.visitpittsburgh.com/events-festivals/food-festivals/", "https://www.picklesburgh.com/",
                       "https://www.pghtacofest.com/", "https://pittsburghrestaurantweek.com/", "https://littleitalydays.com/",
                       "https://bananasplitfest.com/"]
# (list bắt đầu từ 0)link 1, 2, 3 lấy bằng meta_content

website_url_sports_pb = ["https://www.visitpittsburgh.com/things-to-do/pittsburgh-sports-teams/", "https://www.steelers.com/", "https://www.nhl.com/penguins/"]


website_url_info_history_cmu = ["https://www.cmu.edu/about/"]

## Bắt đầu mỗi một text cần một tiêu đề, tạo vòng lặp để lấy tất cả các text từ những link trên rồi cho vào 1 db
#không có chú thích thì lấy bằng hàm craw_text_data_normal



# vector_db_path = "vector_db/db_faiss"

def craw_text_data_normal(website_url):
    response = requests.get(website_url, headers=headers)
    html_content = response.text
    soup = BeautifulSoup(html_content, "html.parser")
    paragraphs = soup.find_all("p")
    text_list = [p.get_text(strip=True) for p in paragraphs]

    # Nối các đoạn văn bản, làm sạch khoảng trắng thừa
    text = " ".join(text_list)

    # các dấu ký tự như \t, \n bị lặp nhiều lần ở đầu mỗi chuỗi (\s+) sẽ được thay thế bằng 1 dấu cách (" ")
    text = re.sub(r"\s+", " ", text).strip()

    # các chuỗi có dạng "[số]" cũng cần được loại bỏ
    text = re.sub(r"\[\d+\]", "", text)
    
    return text

def craw_special_text_data(website_url):
    response = requests.get(website_url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    event_paragraphs = []

    for h1 in soup.find_all("h1"):
        a_tag = h1.find("a", href=True)
        if not a_tag or "/events/event/" not in a_tag["href"]:
            continue

        title = a_tag.get_text(strip=True)
        link = a_tag["href"]

        parent_div = h1.find_parent("div", class_="copyContent")
        if not parent_div:
            continue

        date_div = parent_div.find("div", class_="eventdate")
        datetime = date_div.get_text(strip=True) if date_div else ""

        description = ""
        if date_div and date_div.next_sibling:
            sibling = date_div.next_sibling
            while sibling and isinstance(sibling, str) and sibling.strip() == "":
                sibling = sibling.next_sibling
            if sibling and isinstance(sibling, str):
                description = sibling.strip()

        read_more_link = ""
        read_more = parent_div.find("a", class_="button green right")
        if read_more:
            read_more_link = read_more["href"]

        start_date = ""
        location = ""
        event_url = ""
        json_script = parent_div.find_next("script", type="application/ld+json")
        if json_script:
            try:
                data = json.loads(json_script.string)
                start_date = data.get("startDate", "")
                location = data.get("location", {}).get("address", "")
                event_url = data.get("url", "")
            except json.JSONDecodeError:
                pass

        paragraph = f"""
Event: {title}
Time: {datetime or start_date}
Location: {location if location else 'Unknown'}
Description: {description}
More info: {event_url if event_url else link}
""".strip()
        paragraph = re.sub(r"\t+", " ", paragraph).strip()
        paragraph = re.sub(r"\n+", ";", paragraph).strip()
        event_paragraphs.append(paragraph)

        all_even = "\n\n".join(event_paragraphs)
    return all_even

def get_table_web(soup):
    # Tìm bảng đầu tiên có class wikitable
    table = soup.find("table", {"class": "wikitable sortable"})

    # Trích xuất hàng
    rows = table.find_all("tr")

    # Parse dữ liệu
    data = []
    for row in rows[1:]:  # Bỏ header
        cols = row.find_all("td")
        if len(cols) == 4:
            name = cols[0].get_text(strip=True)
            neighborhood = cols[1].get_text(strip=True)
            typ = cols[2].get_text(strip=True)
            summary = cols[3].get_text(strip=True)
            data.append(f"Name: {name}, Neighborhood: {neighborhood}, Type: {typ}, Summary: {summary}")
    text = ".".join(data)
    return text

def craw_table_text_data(website_url):
    response = requests.get(website_url, headers=headers)
    html_content = response.text
    soup = BeautifulSoup(html_content, "html.parser")
    paragraphs = soup.find_all("p")

    text_list = [p.get_text(strip=True) for p in paragraphs]

    # Nối các đoạn văn bản, làm sạch khoảng trắng thừa
    text = " ".join(text_list)

    # các dấu ký tự như \t, \n bị lặp nhiều lần ở đầu mỗi chuỗi (\s+) sẽ được thay thế bằng 1 dấu cách (" ")
    text = re.sub(r"\s+", " ", text).strip()

    # các chuỗi có dạng "[số]" cũng cần được loại bỏ
    text = re.sub(r"\[\d+\]", "", text)

    table_data = get_table_web(soup)

    text = [text, table_data]

    
    return "\n\n".join(text)

def craw_meta_content_data(website_url):
    response = requests.get(website_url, headers=headers)
    html_content = response.text
    soup = BeautifulSoup(html_content, "html.parser")
    ui_keywords = [
        "viewport", "robots", "theme-color", "charset", "generator",
        "referrer", "google", "apple", "facebook", "twitter","msapplication", "http-equiv",
        "format-detection", "image", "og:type"
    ]

    metas = soup.find_all("meta", attrs={"content": True})
    text_list = []

    for meta in metas:
        meta_name = (meta.get("name") or meta.get("property") or meta.get("http-equiv") or "").lower()

        if any(keyword in meta_name for keyword in ui_keywords):
            continue

        text_list.append(meta["content"])

    text = " ".join(text_list)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\[\d+\]", "", text)

    return text

def get_pdf_data_from_web(website_url):
    url = website_url
    filename = "data/2024_operating_budget.pdf"
    response = requests.get(url, verify=False)
    with open(filename, "wb") as f:
        f.write(response.content)

def craw_text_from_pdf(website_url, folder_data_dir):
    get_pdf_data_from_web(website_url)
    loader = DirectoryLoader(path = folder_data_dir, glob="*.pdf", loader_cls = PyPDFLoader)
    documents = loader.load()
    documents = [doc.model_dump()["page_content"] for doc in documents]
    return "\n".join(documents)


def general_info_history_synthesis(website_url_info_history_pb,
                               pdf_url_info_history_pb,
                               website_url_info_history_cmu):
    general_info_history = "GENERAL DATA AND HISTORY \n\n"

    for url in website_url_info_history_pb:
        text = craw_text_data_normal(url) + "\n"
        general_info_history += text
    
    for url in website_url_info_history_cmu:
        text = craw_text_data_normal(url) + "\n"
        general_info_history += text
    
    for url in pdf_url_info_history_pb:
        text = craw_text_from_pdf(url, data_folder_dir) + "\n"
        general_info_history += text
    
    return general_info_history

def event_synthesis(special_website_url_event_pb,
                    website_url_event_pb):
    event = "EVENT \n\n"

    for url in special_website_url_event_pb:
        text = craw_special_text_data(url) + "\n"
        event += text
    
    for url in website_url_event_pb:
        text = craw_text_data_normal(url) + "\n"
        event += text
    
    return event

def music_culture_synthesis(website_url_music_pb,
                            website_url_museums_pb,
                            website_url_food_pb):
    music = "MUSIC\n\n"
    museums = "MUSEUMS\n\n"
    food = "FOOD\n\n"

    for url in website_url_music_pb:
        text = craw_text_data_normal(url) + "\n"
        music += text

    for i in range(len(website_url_museums_pb)-1):
        text = craw_text_data_normal(website_url_museums_pb[i]) + "\n"
        museums += text
    add = craw_table_text_data(website_url_museums_pb[-1]) + "\n"
    museums += add

    for i in range(len(website_url_food_pb)):
        if i == 1 or i == 2 or i == 3:
            text = craw_meta_content_data(website_url_food_pb[i]) + "\n"
            food += text
        else:
            text = craw_text_data_normal(website_url_food_pb[i]) + "\n"
            food += text
    
    return music + museums + food

def sport_synthesis(website_url_sports_pb):
    sport = "SPORT\n\n"
    for url in website_url_sports_pb:
        text = craw_text_data_normal(url) + "\n"
        sport += text
    return sport
    
def create_vector_db(text_data, vector_db_path):
    text_splitter = CharacterTextSplitter(
        separator=".",# chia văn bản thành phần nhỏ hơn trước khi tạo chunk
        chunk_size = 500, # giới hạn token của mỗi chunk (mỗi chunk có bao nhiêu ký tự)
        chunk_overlap = 50, # chunk sau overlap với chunk trước
        length_function = len # hàm xác định độ dài là hàm đếm ký tự của văn bản
    )
    chunks = text_splitter.split_text(text_data)

    # embedding
    embedding_model = GPT4AllEmbeddings(model_file="models/all-MiniLM-L6-v2-f16.gguf")

    # đưa vào faiss db
    db = FAISS.from_texts(texts=chunks, embedding=embedding_model)
    db.save_local(vector_db_path)
    return db

def create_db(website_url_info_history_pb, pdf_url_info_history_pb,
              special_website_url_event_pb, website_url_event_pb, website_url_music_pb,
              website_url_museums_pb, website_url_food_pb, website_url_sports_pb, website_url_info_history_cmu):

    # general_info_history
    general_info_history_text = general_info_history_synthesis(website_url_info_history_pb,
                                                          pdf_url_info_history_pb,
                                                          website_url_info_history_cmu)
    # event in pb(Pittsburgh)
    event_pittsburgh = event_synthesis(special_website_url_event_pb, website_url_event_pb)

    music_culture = music_culture_synthesis(website_url_music_pb,
                                            website_url_museums_pb,
                                            website_url_food_pb)
    sport = sport_synthesis(website_url_sports_pb)

    create_vector_db(general_info_history_text, "vector_db/info_history_db_faiss")
    create_vector_db(event_pittsburgh, "vector_db/event_pittsburgh_db_faiss")
    create_vector_db(music_culture, "vector_db/music_culture_db_faiss")
    create_vector_db(sport, "vector_db/sport_db_faiss")

    return

create_db(website_url_info_history_pb, pdf_url_info_history_pb,
          special_website_url_event_pb, website_url_event_pb, website_url_music_pb,
          website_url_museums_pb, website_url_food_pb, website_url_sports_pb, website_url_info_history_cmu)







