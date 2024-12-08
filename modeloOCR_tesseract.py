from PIL import Image
import pytesseract
import os

# Configura el path de Tesseract si es necesario (en Windows)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Carpeta para guardar las imagenes procesadas
def procesar_imagenes(images, OCR_text_folder="texto_OCR_extraido"):
    texto_extraido = ""
    resultados = {}  # Diccionario para almacenar resultados por image

    # Crear la carpeta de salida si no existe
    if not os.path.exists(OCR_text_folder):
        os.makedirs(OCR_text_folder)


    for image in images:
        if image.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff')):
            try:
                # Abrir la imagen desde la ruta de la imagen y guardarla en la carpeta image_folder
                #ruta_img = os.path.join(image_folder, image)
                img = Image.open(image)
                imagen = img.convert("RGB")
                
                # Extraer texto con Tesseract
                texto = pytesseract.image_to_string(imagen, lang='spa').strip()
                
                # Guardar en resultados
                resultados[image] = texto
                texto_extraido += texto + "\n\n"
                
                # Guardar texto en un archivo .txt para evaluar la extracción
                
                nombre_imagen_txt = os.path.basename(image).rsplit('.', 1)[0] + ".txt"
                ruta_txt = os.path.join(OCR_text_folder, nombre_imagen_txt)

            except Exception as e:
                print(f"Error procesando la imagen {image}: {e}")
                resultados[image] = f"Error: {str(e)}"
        else:
            resultados[image] = "Formato no válido"
    


    with open(ruta_txt, "w", encoding="utf-8") as image_txt:
                    image_txt.write(texto_extraido)

    print(f"Texto extraído guardado en: {ruta_txt}")
    return {"texto_combinado": texto_extraido, "resultados_por_image": resultados}
