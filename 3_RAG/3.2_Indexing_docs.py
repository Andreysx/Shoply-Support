from langchain_huggingface import HuggingFaceEmbeddings
embed_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vector = embed_model.embed_query("Пример текста для эмбеддинга от HugginFace")
print(len(vector), vector[:5])

# print(__file__)