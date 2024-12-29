import re
from transformers import T5Tokenizer, T5ForConditionalGeneration
import os
from datetime import datetime

# Cargar el modelo y el tokenizador
model_name = "google/flan-t5-large"
tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
model = T5ForConditionalGeneration.from_pretrained(model_name)


# Crear carpeta para guardar el texto limpio
clean_text_folder = "texto_limpio"
if not os.path.exists(clean_text_folder):
    os.makedirs(clean_text_folder)

# Fecha y hora actual para nombres únicos de archivos
fecha_hora_actual = datetime.now()
formato_personalizado = fecha_hora_actual.strftime("%Y%m%d_%H%M%S")


def limpiar_texto(texto):
    print("Limpiando texto...")
    texto = re.sub(r'  ', ' ', texto)
    texto = texto.strip()
    return texto


def dividir_en_secciones(texto):
    """
    Divide el texto completo en secciones utilizando los títulos principales.
    """
    print("Dividiendo texto en secciones...")
    secciones = re.split(r"(\d[.,]\s*(?:Q|P|C|E|Con|Cont)[^\n]*)", texto)
    secciones = [s.strip() for s in secciones if s.strip()]

    secciones_dict = {}
    for i in range(0, len(secciones), 2):
        titulo = secciones[i] if i < len(secciones) else None
        contenido = secciones[i + 1] if i + 1 < len(secciones) else ""
        secciones_dict[titulo] = contenido.strip()
        print(f"TITULO {titulo}")
    return secciones_dict


def procesar_seccion(titulo, contenido):
    """
    Procesa una sección específica y extrae la información correspondiente.
    """
    print(f"Procesando sección: {titulo}")
    if not contenido:
        return {"Error": "Sección vacía"}

    prompt = f"""
    Aquí tienes un prospecto médico extraído por OCR:
    {contenido}

    Por favor, extrae la siguiente información y devuélvela en este formato:
    - Nombre del medicamento: [Nombre del medicamento].
    - Principio activo: [Principio activo del medicamento].
    - Dosis recomendada: [Dosis recomendada para el medicamento].
    - Posibles efectos adversos: [Lista de posibles efectos adversos].

    Si no encuentras información en alguna categoría, déjalo vacío.
    """
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
    #inputs = tokenizer(prompt, return_tensors="pt",max_length=16384, truncation=True,padding=True)
    outputs = model.generate(**inputs, max_length=1024, num_beams=4)
    respuesta_texto = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Respuesta texto: {respuesta_texto}")
    return extraer_informacion(respuesta_texto)


def extraer_informacion(texto):
    print("Extrayendo información...")
    info = {
        "Nombre del medicamento": "",
        "Principio activo": "",
        "Dosis recomendada": "",
        "Posibles efectos adversos": {
            "Muy frecuentes": "",
            "Frecuentes": "",
            "Raros": "",
            "Frecuencia no conocida": ""
        }
    }

    # Buscar el nombre del medicamento
    match_nombre = re.search(r"\s*Qué\s+es\s+(.*?)\s+y\s+para\s+qué\s+se\s+utiliza", texto, re.IGNORECASE)
    if match_nombre:
        print(f"MATCH NOMBRE")
        info["Nombre del medicamento"] = match_nombre.group(1).strip()

    # Buscar el principio activo
    match_principio = re.search(r"El principio activo es\s+(.*?)\s", texto, re.IGNORECASE)
    if match_principio:
        print(f"MATCH PRINCIPIO ACTIVO")
        info["Principio activo"] = match_principio.group(1).strip()

    # Buscar la dosis recomendada
    match_dosis = re.search(r"(?:La dosis recomendada es:\s*|Posología\s*|Posologa\s*)(.*?)(?=\n\d[.,]|\n[A-Z]|$)", texto, re.IGNORECASE | re.DOTALL)
    if match_dosis:
        print(f"MATCH DOSIS")
        info["Dosis recomendada"] = match_dosis.group(1).strip()

    # Buscar efectos adversos por categorías
    categorias = ["Muy frecuentes", "Frecuentes","Poco frecuentes", "Raros", "Muy raros", "Frecuencia no conocida", "Frecuencia desconocida" ]
    for categoria in categorias:
        match_efectos = re.search(
            rf"{categoria}.*?:\s*-\s*(.*?)(?=\s*(Muy frecuentes|Frecuentes|Poco Frecuentes|Raros|Muy raros|Frecuencia no conocida|Frecuencia desconocida|$))",
            texto, re.DOTALL
        )
        if match_efectos:
            print(f"MATCH EFECTOS {categoria}")
            info["Posibles efectos adversos"][categoria] = match_efectos.group(1).strip()
        else:
            info["Posibles efectos adversos"][categoria] = ""

    return info

def consolidar_resultados(resultados):
    print("Consolidando resultados...")

    # Consolidar nombre, principio activo y dosis
    nombre = " ".join([r.get("Nombre del medicamento", "") for r in resultados if r.get("Nombre del medicamento")]).strip()
    principio_activo = " ".join([r.get("Principio activo", "") for r in resultados if r.get("Principio activo")]).strip()
    dosis = " ".join([r.get("Dosis recomendada", "") for r in resultados if r.get("Dosis recomendada")]).strip()

    # Consolidar efectos adversos por categorías

    efectos_adversos = {
        "Muy frecuentes": [],
        "Frecuentes": [],
        "Poco frecuentes": [],
        "Raros": [],
        "Muy raros": [],
        "Frecuencia no conocida": [],
        "Frecuencia desconocida": []
    }
    for r in resultados:
        posibles_efectos = r.get("Posibles efectos adversos", {})
        if isinstance(posibles_efectos, dict):  # Verificar si es un diccionario de categorías
            for categoria, efectos in posibles_efectos.items():
                if efectos:
                    efectos_adversos[categoria].append(efectos)
                    print(f"EFECTOS DE {categoria}: {efectos}")


    # Combinar efectos adversos en cadenas
    efectos_consolidados = {
        categoria: ", ".join(efectos) for categoria, efectos in efectos_adversos.items()
    }

    return {
        "resultado": {
            "Nombre del medicamento": nombre,
            "Principio activo": principio_activo.capitalize(),
            "Dosis recomendada": dosis,
            "Posibles efectos adversos": efectos_consolidados
        }
    }


def procesar_texto_completo(texto):
    print("Procesando texto completo...")
    texto_limpio = limpiar_texto(texto)
    secciones = dividir_en_secciones(texto_limpio)
    resultados = [procesar_seccion(titulo, contenido) for titulo, contenido in secciones.items()]
    return consolidar_resultados(resultados)