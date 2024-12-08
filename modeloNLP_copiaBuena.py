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

# Función para limpiar el texto
def limpiar_texto(texto):
    print("Limpiando texto...")
    texto = re.sub(r'\(.*?\)', '', texto)  # Eliminar paréntesis y su contenido
    #texto = re.sub(r'[^\w\sáéíóúüñÁÉÍÓÚÑ.,;:()¿?¡!/-]', '', texto)  # Eliminar caracteres no deseados
    #texto = re.sub(r'(?<=\w)\.(?=\s)', '.', texto)  # Ajustar puntos al final de palabras
    #texto = re.sub(r'\b\d+\b', '', texto)  # Eliminar números aislados
    texto = re.sub(r'  ', ' ', texto)  # Reducir espacios múltiples a uno solo

    # Guardar el texto limpio en un archivo para depuración
    nombre_imagen_txt = f"texto_limpio_{formato_personalizado}.txt"
    ruta_txt = os.path.join(clean_text_folder, nombre_imagen_txt)
    with open(ruta_txt, "w", encoding="utf-8") as file:
        file.write(texto)

    return texto


# Función para dividir el texto en fragmentos
def dividir_texto(texto, max_tokens=512):
    """
    Divide el texto en fragmentos que no excedan el número máximo de tokens permitidos.
    """
    print("Dividiendo texto en fragmentos...")
    palabras = texto.split()
    fragmentos = []
    fragmento_actual = []

    for palabra in palabras:
        input_ids = tokenizer(" ".join(fragmento_actual + [palabra]))["input_ids"]
        if len(input_ids) <= max_tokens:
            fragmento_actual.append(palabra)
        else:
            fragmentos.append(" ".join(fragmento_actual))
            fragmento_actual = [palabra]

    if fragmento_actual:
        fragmentos.append(" ".join(fragmento_actual))

    return fragmentos


# Procesar un fragmento con Flan-T5
def procesar_fragmento(fragmento):
    print("Procesando fragmento...")
    if not fragmento:
        return {"Error": "Fragmento vacío"}

    # Prompt mejorado para guiar al modelo
    prompt = f"""
    Aquí tienes un prospecto médico extraído por OCR:
    {fragmento}

    Por favor, extrae la siguiente información y devuélvela en este formato:
    - Nombre del medicamento: [Nombre del medicamento].
    - Principio activo: [Principio activo del medicamento].
    - Dosis recomendada: [Dosis recomendada para el medicamento].
    - Posibles efectos adversos: [Lista de posibles efectos adversos].

    Si no encuentras información en alguna categoría, dejalo vacío.
    """
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model.generate(**inputs, max_length=256, num_beams=4)
    respuesta_texto = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"respuesta_texto: {respuesta_texto}")
    return extraer_informacion(respuesta_texto)


# Extraer información estructurada
def extraer_informacion(texto):
    print("Extrayendo información...")
    info = {
        "Nombre del medicamento": "",
        "Principio activo": "",
        "Dosis recomendada": "",
        "Posibles efectos adversos": ""
    }

    # Buscar el nombre del medicamento identificado en la sección 1 de los propspectos médicos por 1. Qué es [nombre] y para que se utiliza
    match_nombre = re.search(r"\s*Qué\s+es\s+(.*?)\s+y\s+para\s+qué\s+se\s+utiliza", texto, re.IGNORECASE)
    if match_nombre:
        info["Nombre del medicamento"] = match_nombre.group(1).strip()
    else:
        info["Nombre del medicamento"] = ""

    
    # Buscar el principio activo del medicamento identificado en cualquier parte del documento las palabras "Principio activo"
    match_principio = re.search(r"\s*principio activo es\s*(.*?)(?=\n|$)", texto, re.IGNORECASE)

    if match_principio:
        info["Principio activo"] = match_principio.group(1).strip()
    else:
        info["Principio activo"] = ""

    # Buscar la dosis recomendada del medicamento identificado en la sección 3 de los propspectos médicos por 3. Cómo tomar [nombre]
    match_dosis = re.search(r"\s*Cómo\s+tomar\s+(.*?)\s*(.*?)(?=\n\s*\d+\.|$)", texto, re.IGNORECASE)
    if match_dosis:
        info["Dosis recomendada"] = match_dosis.group(1).strip()
    else:
        info["Dosis recomendada"] = ""

    # Buscar los efectos adversos del medicamento identificado en la sección 4 de los propspectos médicos por 4. Posibles efectos adversos
    match_efectos = re.search(r"\s*Posibles\s+efectos\s+adversos\s*(.*?)(?=\n\s*\d+\.|$)", texto, re.IGNORECASE)
    if match_efectos:
        info["Posibles efectos adversos"] = match_efectos.group(1).strip()
    else:
        info["Posibles efectos adversos"] = ""

    return info


# Consolidar los resultados
def consolidar_resultados(resultados):
    print("Consolidando resultados...")
    nombre = " ".join([r.get("Nombre del medicamento", "") for r in resultados if r.get("Nombre del medicamento")]).strip()
    principio_activo = " ".join([r.get("Principio activo", "") for r in resultados if r.get("Principio activo")]).strip()
    dosis = " ".join([r.get("Dosis recomendada", "") for r in resultados if r.get("Dosis recomendada")]).strip()
    efectos = " ".join([r.get("Posibles efectos adversos", "") for r in resultados if r.get("Posibles efectos adversos")]).strip()

    return {
        "resultado": {
            "Nombre": nombre,
            "Principio activo": principio_activo,
            "Dosis recomendada": dosis,
            "Posibles efectos adversos": efectos
        }
    }


# Procesar texto completo
def procesar_texto_completo(texto):
    
    print("Procesando texto completo...")
    texto_limpio = limpiar_texto(texto)
    fragmentos = dividir_texto(texto_limpio, max_tokens=512)
    resultados = [procesar_fragmento(fragmento) for fragmento in fragmentos]
    return consolidar_resultados(resultados)
