import os
from flask import Flask, jsonify, request
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_elasticsearch import ElasticsearchStore
from psycopg_pool import ConnectionPool #Clase para manejar un pool de conexiones a PostreSQL, eficiente para múltiples accesos concurrentes
from langgraph.checkpoint.postgres import PostgresSaver #Permite guardar y restaurar checkpoints en PostgreSQL
from langgraph.prebuilt import create_react_agent #para crear el agente
#Agregados para la gestión de la memoria a corto plazo
from langmem.short_term import SummarizationNode,RunningSummary #SummarizationNode: nodo que resume cuando se pasa el umbral de tokens | #RunningSummary: clase para almacenar y actualizar el resumen progresivamente.
from langchain_core.messages.utils import count_tokens_approximately #Función que estima cuántos tokens tiene un mensaje. Se usa para decidir cuándo resumir la conversación.
from langgraph.prebuilt.chat_agent_executor import AgentState #es la clase base del estado del agente (mensajes, acciones, etc.).
from typing import Optional #Importa anotación de tipos, usada para definir estructuras como dict[str, RunningSummary].
from langgraph.graph import StateGraph, START, MessagesState,add_messages
#StateGraph, defines nodos y transiciones entre ellos, como en una máquina de estados finitos.
#START, Constante especial que representa el punto de entrada del grafo
#MessagesState, Una estructura de estado predefinida que representa una conversación como una lista de mensajes
from langgraph.graph.message import AnyMessage,add_messages,RemoveMessage
from langchain.tools import tool,StructuredTool

from langchain.pydantic_v1 import BaseModel, Field

from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_community.agent_toolkits import SQLDatabaseToolkit


index_url = os.getenv("index_url")
index_name = os.getenv("index_name")
langchain_password= os.getenv("langchain_password")
uribd =os.getenv("uribd")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

os.environ["LANGSMITH_ENDPOINT"]="https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "gcpaiagent"

app = Flask(__name__)

# ==================== TOOL 1: RAG PARA INFORMACIÓN GENERAL DE PRIMA AFP ====================
class PrimaAFPInfoInput(BaseModel):
    """Esquema para búsqueda de información general sobre Prima AFP"""
    query: str = Field(description="Consulta sobre información general de Prima AFP") # Consulta obligatoria
    top_k: int = Field(default=5, description="Número de resultados (máximo 10)") # Número de resultados (máx 10)
    categoria: Optional[str] = Field(default=None, description="Categoría: 'descripcion','fondos', 'traslados', 'programas', 'inversiones', 'general', 'Ahorro Ya','Tipos de fondo','Programa AhorroYa!' ") # Filtro por categoría
    min_density: Optional[float] = Field(default=0.3, description="Densidad semántica mínima para filtrar resultados") # Filtro por densidad semántica

def buscar_info_general_prima_afp(
    query: str,
    top_k: int = 5,
    categoria: Optional[str] = None,
    min_density: Optional[float] = 0.3
) -> str:
    """
    Función de búsqueda RAG para información general de Prima AFP (categoría no estricta)
    """
    try:
        if top_k > 10:
            top_k = 10
        if top_k < 1:
            top_k = 1
            
        # Conexión a Elasticsearch
        embeddings = OpenAIEmbeddings()
        
        db = ElasticsearchStore(
            embedding=embeddings,
            index_name=index_name,
            es_url=index_url,
            es_user="elastic",
            es_password=langchain_password,
        )
        
        # Búsqueda inicial
        initial_results = min(top_k * 5, 50)
        docs = db.similarity_search(query, k=initial_results)
        #print(docs)
        # Separar por prioridad: con categoría y sin categoría
        priorizados = []
        secundarios = []
        
        for doc in docs:
            metadata = doc.metadata
            density = metadata.get('semantic_density', 0)
            
            if density < min_density:
                continue  # filtrar por densidad
            
            categoria_doc = metadata.get("categoria", "").lower()
            if categoria and categoria.lower() in categoria_doc:
                priorizados.append(doc)
            else:
                secundarios.append(doc)
            
            if len(priorizados) >= top_k:
                break
        
        # Completar con secundarios si faltan documentos
        total_docs = priorizados + secundarios
        filtered_docs = total_docs[:top_k]
        
        # Formatear respuesta
        if not filtered_docs:
            return f"No encontré información sobre '{query}'. ¿Podrías ser más específico?"
        
        response_parts = [f"📚 Información sobre '{query}':\n"]
        
        for i, doc in enumerate(filtered_docs):
            response_parts.append(f"📋 INFORMACIÓN {i+1}:")
            
            metadata = doc.metadata
            categoria_info = metadata.get('categoria', 'Información general')
            response_parts.append(f"   📂 Categoría: {categoria_info}")
            
            content = doc.page_content.strip().replace("\n", " ")
            response_parts.append(f"   ℹ️  Contenido: {content[:400]}...")
            
            if metadata.get('semantic_density', 0) > 0.6:
                response_parts.append(f"   ⭐ Información verificada")
            
            response_parts.append("")
        
        return "\n".join(response_parts)
        
    except Exception as e:
        return f"❌ Error al buscar información: {str(e)}"  
# ====================================================================================================#

# OLD TOOL
###

@app.route('/agent', methods=['GET'])
def main():
    id_usuario = request.args.get('idagente')
    msg = request.args.get('msg')

    # ==================== TOOL 1: RAG PARA PREGUNTAS GENERALES===================
    tool_info_general = StructuredTool.from_function(
        func=buscar_info_general_prima_afp, # Función que ejecuta
        name="info_general_prima_afp", # Nombre identificador
        description="""
        Herramienta para buscar información GENERAL sobre Prima AFP usando RAG.
        
        Úsala SOLO para:
        - Información sobre cómo trasladarse a Prima AFP
        - Descripción de los tipos de fondos (características, perfiles de riesgo)
        - Programas de Prima AFP (Ahorro Ya, etc.)
        - Dónde invierte Prima AFP los fondos
        - Información institucional y general
        - Preguntas sobre el sistema de pensiones
        
        NO usar para consultas personalizadas del afiliado (usa las otras herramientas).
        """,
        args_schema=PrimaAFPInfoInput # Esquema de parámetros
    )  
    #====================================================================================================
    
    # ==================== TOOL 2: TOOL DE SQL CONSULTAS PERSONALIZADAS POR AFILIADO ====================

    # Conexión a la base de datos
    db_sql = SQLDatabase.from_uri(
        database_uri=uribd,
        include_tables=[
            "afiliados",
            "saldos_fondos",
            "aportes",
            "tramites",
            "rentabilidad_fondos",
            "empleadores"
        ]
    )

    # Herramienta para consultar la base de datos
    #sql_tool = QuerySQLDataBaseTool(db=db_sql)

    model = ChatOpenAI(model="gpt-4.1-2025-04-14")
    toolkit_bd = SQLDatabaseToolkit(db=db_sql,llm=model)
    sql_tools = toolkit_bd.get_tools()

    #====================================================================================================

############################## DEFINICION DEL AGENTE######################
# Variables de memoria
    connection_kwargs = {
        "autocommit": True,
        "prepare_threshold": 0,
    }

    # Inicializamos la memoria
    with ConnectionPool(
        conninfo=uribd,
        max_size=20,
        kwargs=connection_kwargs,
    ) as pool:

        checkpointer = PostgresSaver(pool)

        # Inicializamos el modelo - CORREGIDO: modelo válido
        model = ChatOpenAI(model="gpt-4.1-2025-04-14") #gpt-4.1-2025-04-14

        summarization_node = SummarizationNode(
            token_counter=count_tokens_approximately,
            model=model,
            max_tokens=4048, #umbral para decirle al agente cuando se debe resumir el contexto que se le envía al modelo
            max_summary_tokens=2048, #límite de tokens del resumen
            output_messages_key="llm_input_messages", #Variable donde se guarda el historial de los mensajes, sobre el cual se decide si resumir o no
        )

        class State(AgentState):
            # NOTE: we're adding this key to keep track of previous summary information
            # to make sure we're not summarizing on every LLM call
            #context: dict[str, Any]
            context: dict[str, RunningSummary]

        #toolkit = [tool_rag]  # era 'tolkit'
        toolkit = [tool_info_general] + sql_tools
        #toolkit = sql_tools

        system_prompt = """
            Eres un asistente gentil y especializado de Prima AFP (una administradora de fondo de pensiones) que puede ayudar tanto con información general como con consultas personalizadas de afiliados.
            
            Tienes varias herramientas disponibles:
            1.- **info_general_prima_afp**: Para información GENERAL sobre Prima AFP
                - Cómo trasladarse a Prima AFP
                - Descripción de fondos (características, perfiles de riesgo)
                - Programas como Ahorro Ya
                - Dónde invierte Prima AFP
                - Información institucional

            2.- Vas a interactuar con una base de datos SQL para resolver las consultas que te hagan
                Dada una pregunta de entrada, crea una consulta {dialect} sintácticamente correcta para ejecutarla, luego analiza los resultados de la consulta y devuelve la respuesta. A menos que el usuario especifique un número específico de ejemplos que desea obtener, limita siempre tu consulta a un máximo de {top_k} resultados.
                Puedes ordenar los resultados por una columna relevante para devolver los ejemplos más interesantes de la base de datos. Nunca consultes todas las columnas de una tabla específica, solo solicita las columnas relevantes dada la pregunta.

                Debes revisar tu consulta antes de ejecutarla. Si recibes un error al ejecutar una consulta, reescríbela e inténtalo de nuevo.

                NO realices ninguna instrucción DML (INSERT, UPDATE, DELETE, DROP, etc.) en la base de datos.

                Para empezar, SIEMPRE debes revisar las tablas de la base de datos para ver qué puedes consultar. NO omitas este paso.

                Luego debes consultar el esquema de las tablas más relevantes.

            Si no cuentas con una herramienta específica para resolver una pregunta, infórmalo claramente e indica cómo puedes ayudar.
            
            Instrucciones:
                - Responde en español con amplios detalles según el contexto proporcionado
                - Si no sabes la respuesta, di claramente que no tienes esa información, no supongas ni alucines.
                - Si encuentras links en el contexto, incluyelos obligatoriamente en la respuesta
                - Usa emojis para organizar la información
                - Siempre pregunta si necesita más información

            """.format(
                dialect=db_sql.dialect,
                top_k=5,
            )

        agent_executor = create_react_agent(
                model,
                tools=toolkit,
                pre_model_hook=summarization_node,
                state_schema=State,
                checkpointer=checkpointer,
                prompt=system_prompt,
            )

        config = {"configurable": {"thread_id": id_usuario}}
        msg_user = msg + "el usuario es: " + id_usuario
        response = agent_executor.invoke({"messages": [HumanMessage(content=msg_user)]}, config=config)
        
        #return jsonify({"msg": "Agente iniciado correctamente"})
        return response['messages'][-1].content

if __name__ == '__main__':
    # La aplicación escucha en el puerto 8080, requerido por Cloud Run
    app.run(host='0.0.0.0', port=8080)    