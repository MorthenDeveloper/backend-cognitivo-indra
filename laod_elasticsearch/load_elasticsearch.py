import pdfplumber
from google.cloud import storage
from io import BytesIO
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import re
import os
from langchain_openai import OpenAIEmbeddings #convertir a embeddings
from langchain.vectorstores import ElasticsearchStore #para conectarnos a elasticsearch

index_url = os.getenv("index_url")
index_name = os.getenv("index_name")
langchain_password= os.getenv("langchain_password")
uribd =os.getenv("uribd")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

chunk_size=1000
chunk_overlap=200

def listar_archivos(bucket_name, prefix=None):
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    blobs = bucket.list_blobs(prefix=prefix)  # `prefix` filtra por carpeta si lo necesitas
    
    archivos = [blob.name for blob in blobs]

    return archivos

def get_text_metadata_new_files(bucket_name: str, new_files: list):
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    all_text_and_metadata = []

    for file_name in new_files:
        blob = bucket.blob(file_name)
        pdf_bytes = blob.download_as_bytes()

        all_text = ''  # Reiniciar el texto por archivo

        with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
            for pdf_page in pdf.pages:
                single_page_text = pdf_page.extract_text()
                if single_page_text:
                    all_text += '\n' + single_page_text

        all_text_and_metadata.append({
            "file_name": file_name,
            "text": all_text.strip(),
        })

    return all_text_and_metadata

def perform_semantic_chunking(document_text: str, chunk_size=1000, chunk_overlap=200):

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    
    semantic_chunks = text_splitter.split_text(document_text)
    
    #Para que reconozca encabezados y metadatos usando expresiones regulares
    section_patterns = [
        r'^\d+\.\s+.+$',         # Títulos numerados ("1. Registro")
        r'^.*:\s*$',             # Mayúsculas con dos puntos ("PROTECCIÓN DE DATOS:")
        r'^#+\s+.+$',            # Markdown (# Título)
        r'^.+\n[=\-]{2,}$'       # Subrayados
    ]

    separators=[
        "\n\n",         # Doble salto de línea
        "\n",           # Salto de línea
        r"\n\d+\.\s",   # Títulos numerados tipo "1. Introducción"
        ". ",           # Punto seguido
        " ",            # Espacios
        ""              # Carácter por carácter como último recurso
    ]
    
    documents = []
    current_section = "Descripción"
    
    #Creamos los Document con metadatos
    for i, chunk in enumerate(semantic_chunks): #Recorremos cada chunk generado
        chunk_lines = chunk.split('\n')
        for line in chunk_lines:
            for pattern in section_patterns:
                match = re.match(pattern, line, re.MULTILINE)
                if match:
                    current_section = match.group(0)
                    break
        
        words = re.findall(r'\b\w+\b', chunk.lower())
        stopwords = ['el', 'y', 'es', 'de', 'a', 'un', 'en', 'que', 'eso', 'con', 'como', 'para']
        content_words = [w for w in words if w not in stopwords]
        semantic_density = len(content_words) / max(1, len(words))
        
        doc = Document(
            page_content=chunk,
            metadata={
                "chunk_id": i,
                "total_chunks": len(semantic_chunks),
                "chunk_size": len(chunk),
                "chunk_type": "semantic",
                "section": current_section,
                "semantic_density": round(semantic_density, 2)
            }
        )
        documents.append(doc)
    
    return documents

def create_embeddings(documents):

    embeddings = OpenAIEmbeddings() #instancia de la clase OpenAIEmbeddings, para transformar en embeddings

    db = ElasticsearchStore.from_documents( #subo la data spliteada a la base de datos vectorial
        documents,
        embeddings,
        es_url=index_url,
        es_user="elastic",
        es_password=langchain_password,
        index_name=index_name,
    )
    
    db.client.indices.refresh(index=index_name)

    print("✅ ¡Chunks cargados exitosamente a Elasticsearch!")

def test_elasticsearch_index(query: str, top_k: int = 3):
    """
    Realiza una búsqueda semántica en el índice de Elasticsearch para probar los embeddings.
    
    Parámetros:
        query (str): La consulta en lenguaje natural.
        top_k (int): Número de resultados más similares a retornar.
    
    Retorna:
        Lista de documentos más similares con su metadata.
    """

    embeddings = OpenAIEmbeddings()
    
    db = ElasticsearchStore(
        embedding=embeddings,
        index_name=index_name,
        es_url=index_url,
        es_user="elastic",
        es_password=langchain_password,
    )

    docs = db.similarity_search(query, k=top_k)

    print(f"🔍 Resultados para la consulta: '{query}'\n")
    for i, doc in enumerate(docs):
        print(f"📄 Documento #{i+1}")
        print(f"Contenido:\n{doc.page_content}...")#{doc.page_content[:300]}
        print(f"Metadata:\n{doc.metadata}")
        print("-" * 60)

    return docs

#######################################################################

def search_with_metadata(query: str, top_k: int = 5, **metadata_filters):
    """
    Búsqueda simple que aprovecha la metadata de los chunks.
    
    Parámetros:
        query (str): La consulta en lenguaje natural
        top_k (int): Número de resultados a retornar
        **metadata_filters: Filtros opcionales de metadata
        
    Ejemplos de uso:
        # Búsqueda normal
        search_with_metadata("programa ahorro")
        
        # Buscar en archivo específico
        search_with_metadata("programa ahorro", file_name="documento.pdf")
        
        # Buscar en sección específica
        search_with_metadata("registro", section="Registro")
        
        # Buscar con densidad semántica mínima
        search_with_metadata("ahorro", min_density=0.3)
    """
    
    embeddings = OpenAIEmbeddings()
    
    db = ElasticsearchStore(
        embedding=embeddings,
        index_name=index_name,
        es_url=index_url,
        es_user="elastic",
        es_password=langchain_password,
    )
    
    # Obtener más resultados de los necesarios para poder filtrar
    initial_results = top_k * 5
    docs = db.similarity_search(query, k=initial_results)
    
    # Aplicar filtros de metadata
    filtered_docs = []
    
    for doc in docs:
        include_doc = True
        metadata = doc.metadata
        
        # Filtros simples y directos
        for filter_key, filter_value in metadata_filters.items():
            
            if filter_key == "file_name":
                if filter_value not in metadata.get('file_name', ''):
                    include_doc = False
                    break
                    
            elif filter_key == "section":
                if filter_value.lower() not in metadata.get('section', '').lower():
                    include_doc = False
                    break
                    
            elif filter_key == "min_density":
                if metadata.get('semantic_density', 0) < filter_value:
                    include_doc = False
                    break
                    
            elif filter_key == "min_size":
                if metadata.get('chunk_size', 0) < filter_value:
                    include_doc = False
                    break
                    
            elif filter_key == "chunk_type":
                if metadata.get('chunk_type') != filter_value:
                    include_doc = False
                    break
        
        if include_doc:
            filtered_docs.append(doc)
            
        # Parar cuando tengamos suficientes resultados
        if len(filtered_docs) >= top_k:
            break
    
    # Mostrar resultados
    print(f"🔍 Consulta: '{query}'")
    if metadata_filters:
        print(f"📋 Filtros aplicados: {metadata_filters}")
    print(f"📊 Resultados encontrados: {len(filtered_docs)}\n")
    
    for i, doc in enumerate(filtered_docs):
        print(f"📄 Resultado #{i+1}")
        print(f"   📁 Archivo: {doc.metadata.get('file_name', 'N/A')}")
        print(f"   📑 Sección: {doc.metadata.get('section', 'N/A')}")
        print(f"   🎯 Densidad: {doc.metadata.get('semantic_density', 'N/A')}")
        print(f"   📏 Tamaño: {doc.metadata.get('chunk_size', 'N/A')} caracteres")
        print(f"   💬 Contenido: {doc.page_content}...\n")
    
    return filtered_docs

# Métodos de conveniencia aún más simples
def search_in_file(query: str, file_name: str, top_k: int = 3):
    """Buscar solo en un archivo específico"""
    return search_with_metadata(query, top_k, file_name=file_name)

def search_in_section(query: str, section: str, top_k: int = 3):
    """Buscar solo en una sección específica"""
    return search_with_metadata(query, top_k, section=section)

def search_high_quality(query: str, top_k: int = 3):
    """Buscar solo en chunks de alta calidad (alta densidad semántica)"""
    return search_with_metadata(query, top_k, min_density=0.4, min_size=200)

#######################################################    


def main():
    archivos = listar_archivos(bucket_name)

    # Filtramos los archivos pdf
    archivos_pdf = [f for f in archivos if f.endswith('.pdf')]

    #Se obtiene una lista de diccionarios
    resultados = get_text_metadata_new_files(bucket_name, archivos_pdf)

    todos_los_chunks = []
    #a la lista de resultados

    for item in resultados:
        file_name = item["file_name"]
        text = item["text"]
        #print("\n" + file_name)
        #print(text+ "\n\n")

        #Aplicamos el chunking semántico al texto
        semantic_chunks = perform_semantic_chunking(text,chunk_size,chunk_overlap)

        # Agrega el nombre del archivo como metadata extra
        for chunk in semantic_chunks:
            chunk.metadata["file_name"] = file_name

        todos_los_chunks.extend(semantic_chunks) #Lista de Documents

    print(f"Total chunks generados: {len(todos_los_chunks)}")

    #for i, doc in enumerate(todos_los_chunks):
        #print(f"chunk" + str(i) + ": " + doc)
        #print(doc.metadata)

    #Creamos los embeddings y los cargamos a Elasticsearch
    create_embeddings(todos_los_chunks)

if __name__ == "__main__":
    #main() #Cargar los documentos como embeddings a ElasticSearch
    query = "Qué es Ahorro Ya"
    #test_elasticsearch_index(query,5)

    # Buscar solo en un archivo
    #search_with_metadata("programa", file_name="documento.pdf")

    ## Buscar solo en una sección
    #search_with_metadata("registro", section="Registro")

    ## Buscar chunks de buena calidad
    #search_with_metadata("ahorro", min_density=0.4, min_size=200)

    # Combinando filtros
    #search_with_metadata("programa", file_name="doc.pdf", min_density=0.3)

    print("\n=== BÚSQUEDA CON MÚLTIPLES FILTROS ===")
    search_with_metadata(
        query, 
        top_k=5,
        min_density=0.3, 
        #min_size=150
    )

