#  Тестовое задание: Индексация документа и поиск по эмбеддингам через FAISS
from langchain_community.document_loaders import PyMuPDFLoader, WebBaseLoader
from langchain_text_splitters import TokenTextSplitter, RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import bs4

# Загрузчики возвращают объекты Document. Каждый объект имеет два основных атрибута:
# page_content – текстовое содержимое (строка)
# metadata – словарь с метаданными (источник, страница и т.д.)

# Загрузка первой страницы
first_html_loader = WebBaseLoader(
    web_path="https://tlc-selyatino.ru/terminal-vostok-zapad/",
    bs_kwargs={
        "parse_only": bs4.SoupStrainer(id="about-branch-block")
    }
 )
first_html_docs = first_html_loader.load()



# Загрузка второй страницы
second_html_loader = WebBaseLoader(
    web_path="https://tlc-selyatino.ru/impeks-tb/",
    bs_kwargs={
        "parse_only": bs4.SoupStrainer(id="about-branch-block")
    }
 )
second_html_docs = second_html_loader.load()

# Загрузка третьей страницы
html_loader = WebBaseLoader(
    web_paths=("https://docs.langchain.com/oss/python/langchain/overview",),
    bs_kwargs={
        "parse_only": bs4.SoupStrainer(id="content")
    }
)
html_docs = html_loader.load()



pdf_loader = PyMuPDFLoader(
    file_path="test.pdf",
    # mode="page",  # "page" или "single"
    # extract_images=False,  # извлечение изображений
    # extract_tables="markdown",  # извлечение таблиц в формате markdown
)

pdf_docs = pdf_loader.load()

#Объединение всех документов
all_docs = pdf_docs + html_docs + second_html_docs + first_html_docs
print(f"Всего документов: {len(all_docs)}")
# Просмотр структуры
# for i, doc in enumerate(all_docs):  # первые 3 документа
#     print(f"\n--- Документ {i+1} ---")
#     print(f"Источник: {doc.metadata.get('source', 'N/A')}")
#     print(f"Первые 150 символов: {doc.page_content[:150]}...")


# Разбиение на чанки(фрагменты) по токенам при помощи TokenTextSplitter или RecursiveCharacterTextSplitter
# text_splitter = TokenTextSplitter(encoding_name="cl100k_base", chunk_size=200, chunk_overlap=20)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
splitted_docs = text_splitter.split_documents(all_docs)
print(f"Было документов: {len(all_docs)}, стало фрагментов: {len(splitted_docs)}")
# for i, doc in enumerate(splitted_docs):  # первые 3 документа
#     print(f"\n--- Документ {i+1} ---")
#     print(f"Источник: {doc.metadata.get('source', 'N/A')}")
#     print(f"Первые 150 символов: {doc.page_content}")

#Создание модели эмбеддингов, локальная модель для создания векторов с HugginFace
embed_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

#  Создание векторного хранилища из  чанков FAISS (Facebook AI Similarity Search),  указав модель эмбедингов
vector_store = FAISS.from_documents(splitted_docs, embed_model)
vector_store.save_local("index/my_faiss_index")


def query_input(query: str):
    new_store = FAISS.load_local("index/my_faiss_index", embed_model, allow_dangerous_deserialization=True)
    
    result = new_store.similarity_search(query, k=2)
    print("Необходимый фрагмент(чанк) текста: ")
    print()
    print(result[0].page_content)
    print("Metadata к чанку: ")
    print(result[0].metadata)
    
while True:
    user_input = input("Введите ваш запрос: ")
    if user_input in ['стоп', 'остановить', 'все']:
        print("До свидания!")
        break
    else:
        query_input(user_input)
    