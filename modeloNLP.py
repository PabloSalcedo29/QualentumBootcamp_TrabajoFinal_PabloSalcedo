import regex as re
DEVICE = "cpu"

from transformers import AutoModelForCausalLM, AutoTokenizer

nlp_model = "Qwen/Qwen1.5-1.8B" #modelo empleado para extraer la infromación deseada del texto extraído
#nlp_model = "Qwen/Qwen2.5-1.5B-Instruct" #modelo empleado para extraer la infromación deseada del texto extraído
#nlp_model = "microsoft/MiniLM-L12-H384-uncased" #modelo empleado para extraer la infromación deseada del texto extraído
#nlp_model = "sentence-transformers/all-MiniLM-L6-v2" #modelo empleado para extraer la infromación deseada del texto extraído
#nlp_model = "google/flan-t5-small" #modelo empleado para extraer la infromación deseada del texto extraído

model = AutoModelForCausalLM.from_pretrained(nlp_model).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(nlp_model)


def limpiar_texto(texto):
        # Eliminar caracteres no deseados como /, \, *, etc.
        texto = re.sub(r"[\/\\\*\|\"<>().]", "", texto)
        # Remover espacios en blanco al inicio y al final, y espacios duplicados dentro del texto
        texto = re.sub(r"\s+", " ", texto).strip()
        return texto

def pipeline_completo(prospecto_médico: str, model, tokenizer) -> dict[str, dict[str, str]]:

       #prompt que indica al modelo nlp_model que información queremos que extraiga
    prompt = (
            f"Extrae el nombre del medicamento, la dosis, la frecuencia de toma y una lista de posinles efectos secundarios separados por comas del siguiente prospecto médico.\n\n"
            f"texto: {prospecto_médico}\n\n"
            f"Quiero la información en el siguiente formato:\n"
            f"Nombre: \n"
            f"Dosis: \n"
            f"Frecuencia de toma: \n"
            f"Posibles efectos secundarios: \n"
            f"Si algún campo no está presente, deja el campo vacío (\"\")."
        )

        # Tokenización y preparación del texto para el modelo
    inputs = tokenizer(prompt, return_tensors="pt")

      # Generación de la respuesta usando el modelo
    outputs = model.generate(**inputs, max_length=500, temperature=0.7, top_p=0.9, repetition_penalty=1.1)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

      # Inicialización del diccionario de resultado
    respuesta = {
          "Nombre": "",
          "Dosis": "",
          "Frecuencia de toma": "",
          "Posibles efectos secundarios": ""
      }

      # Procesar cada línea y extraer los campos esperados
    for linea in result.split("\n"):
          if "Título:" in linea:
              respuesta["Título"] = limpiar_texto(linea.replace("Título:", "").strip())
          elif "Director:" in linea:
              respuesta["Director"] = limpiar_texto(linea.replace("Director:", "").strip())
          elif "Año estreno:" in linea:
              respuesta["Año estreno"] = limpiar_texto(linea.replace("Año estreno:", "").strip())

    return respuesta
