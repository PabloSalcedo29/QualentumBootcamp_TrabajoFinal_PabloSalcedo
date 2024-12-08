from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
from modeloOCR_tesseract import procesar_imagenes
from modeloNLP import procesar_texto_completo

app = Flask(__name__)

UPLOAD_FOLDER = 'imagenes_prospectos_medicos'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MAX_FILE_SIZE_MB = 5  # Límite de tamaño de archivo en MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('interfazWeb.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No se encontró ningún archivo"}), 400

    archivos = request.files.getlist('file')

    if not archivos:
        return jsonify({"error": "No se seleccionaron archivos"}), 400

    archivos_guardados = []
    for archivo in archivos:
        if archivo.filename == '':
            return jsonify({"error": "Uno o más archivos no tienen nombre"}), 400

        if archivo and allowed_file(archivo.filename):
            # Verificar el tamaño del archivo
            archivo.seek(0, os.SEEK_END)
            file_size_mb = archivo.tell() / (1024 * 1024)
            archivo.seek(0)
            if file_size_mb > MAX_FILE_SIZE_MB:
                return jsonify({"error": f"Archivo demasiado grande: {archivo.filename}"}), 400

            filename = secure_filename(archivo.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            archivo.save(file_path)
            archivos_guardados.append(file_path)
        else:
            return jsonify({"error": f"Archivo con formato no válido: {archivo.filename}"}), 400

    try:
        # Procesar con OCR
        resultado_ocr = procesar_imagenes(archivos_guardados)
        texto_extraido = resultado_ocr["texto_combinado"]
        print("ocr done")
        #print(f"Texto extraído del OCR:\n{texto_extraido}")

        if not texto_extraido.strip():
            raise ValueError("No se pudo extraer texto válido del prospecto.")
    except Exception as e:
        return jsonify({"error": f"Error al procesar las imágenes con OCR: {str(e)}"}), 500

    try:
        resultado_nlp = procesar_texto_completo(texto_extraido)
        nombre_medicamento = resultado_nlp.get("resultado", {}).get("Nombre", "Información no disponible")
        principio_activo = resultado_nlp.get("resultado", {}).get("Principio activo", "Información no disponible")
        dosis_recomendada = resultado_nlp.get("resultado", {}).get("Dosis recomendada", "Información no disponible")
        posibles_efectos = resultado_nlp.get("resultado", {}).get("Posibles efectos", "Información no disponible")
    except Exception as e:
        return jsonify({"error": f"Error al procesar el texto con el modelo NLP: {str(e)}"}), 500

    return jsonify({
        "status": "success",
        "nombre_medicamento": nombre_medicamento,
        "principio_activo": principio_activo,
        "dosis_recomendada": dosis_recomendada,
        "posibles_efectos": posibles_efectos
})


if __name__ == '__main__':
    app.run(debug=True)
