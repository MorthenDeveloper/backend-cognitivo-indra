# backend-cognitivo-indra
🤖 Prima AFP - Agente Inteligente Multitarea
Este proyecto implementa un agente inteligente para Prima AFP, capaz de responder tanto preguntas generales sobre el sistema de pensiones como consultas personalizadas de afiliados mediante integración con Elasticsearch (RAG) y una base de datos PostgreSQL. Se construye sobre LangChain, Flask, y modelos de OpenAI.

## 📁 Estructura del proyecto

```plaintext
├── agente/
│   ├── app.py                # Archivo principal del servidor Flask y configuración del agente.
│   ├── Dockerfile            # Dockerfile para desplegar la app
│   └── requirements.txt      # Dependencias necesarias para la app
│
├── load_elasticsearch/
│   └── load_elasticsearch.py # Script para procesar, chunkear e indexar los PDF a Elasticsearch
│
├── docs/
│   └── Arquitectura AGENTE GCP.png  # Diagrama arquitectónico del proyecto en GCP
```

> Explicación proyecto: https://youtu.be/61XnuW8QvkE

🚀 Funcionalidades principales
✅ 1. Búsqueda General (RAG)
Se implementa una herramienta llamada info_general_prima_afp basada en Elasticsearch + embeddings de OpenAI para responder consultas generales sobre:

Tipos de fondo

Traslado de AFP

Programas como Ahorro Ya

Dónde invierte Prima AFP

Información institucional y del sistema de pensiones

✅ 2. Consultas SQL personalizadas
El agente puede conectarse a una base de datos PostgreSQL con múltiples tablas (afiliados, aportes, saldos_fondos, etc.), y ejecutar consultas con validación automática. Usa LangChain SQLDatabaseToolkit

🧠 Arquitectura del Agente
Modelo LLM: GPT-4.1 vía ChatOpenAI

Memoria a corto plazo: Se gestiona usando RunningSummary y SummarizationNode para resumir conversaciones largas si exceden el límite de tokens.

Persistencia de estado: Usa PostgresSaver para guardar el estado y el historial del agente en PostgreSQL.

Estado personalizado: Define una clase State que hereda de AgentState e incluye el resumen acumulado de la conversación.

Grafo de estados (StateGraph): El agente usa un grafo para decidir cuándo ejecutar herramientas, resumir el estado o generar una respuesta.

⚙️ Variables de entorno requeridas (.env)
env
Copiar
Editar
OPENAI_API_KEY=tu_clave_openai
LANGCHAIN_API_KEY=tu_clave_langchain
LANGCHAIN_PROJECT=gcpaiagent
LANGCHAIN_TRACING_V2=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com

index_url=http://localhost:9200  # o URL pública
index_name=nombre_del_indice_elasticsearch
langchain_password=clave_elasticsearch
uribd=postgresql://usuario:password@host:puerto/nombrebd

🛠️ Endpoints disponibles
/agent – GET
Permite enviar un mensaje a un agente determinado.

| Parámetro  | Descripción                        | Tipo  |
| ---------- | ---------------------------------- | ----- |
| `idagente` | Identificador del agente o usuario | `str` |
| `msg`      | Pregunta o mensaje del usuario     | `str` |

Ejemplo:
GET /agent?idagente=123&msg=¿Cómo me traslado a Prima AFP?

🧠 Memoria y resumen conversacional
Se usa un resumen progresivo de tokens para evitar desbordes de contexto (SummarizationNode, RunningSummary).

El resumen se guarda junto con el estado del agente en PostgreSQL usando PostgresSaver.

💬 Prompt de sistema
El agente recibe instrucciones claras para:

Usar herramientas cuando estén disponibles

No alucinar si no tiene información

Mostrar resultados con emojis para facilitar la lectura

Responder en español con contexto

📌 Tecnologías usadas
LangChain

- Flask
- OpenAI GPT-4.1
- Elasticsearch + RAG
- PostgreSQL + psycopg_pool
- LangGraph
- LangMem

✉️ Contacto
Desarrollado por el equipo de IA y Datos en Prima AFP
📫 Contacto: [fabianvillanuevajaqui@gmail.com]