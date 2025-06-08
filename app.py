# app.py - Aplicación de Desenfoque Automático de Caras

import os
import cv2
import torch
import numpy as np
import gradio as gr
from pathlib import Path
import time
from typing import Tuple, List

# Las dependencias se instalan a través de requirements.txt en Hugging Face Spaces
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

class FaceBlurApp:
    def __init__(self):
        """Inicializa la aplicación y los modelos."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🔧 Usando dispositivo: {self.device}")
        
        # Inicializar modelos
        self.init_models()
        
        # Estadísticas de rendimiento
        self.reset_stats()
    
    def init_models(self):
        """Inicializa los modelos YOLO y DeepSORT."""
        try:
            # Cargar modelo YOLOv8 (más estable que YOLOv9)
            # YOLO descargará los pesos la primera vez que se ejecute.
            self.yolo_model = YOLO('yolov8n.pt')
            print("✅ Modelo YOLO cargado")
            
            # Inicializar DeepSORT
            self.tracker = DeepSort(max_age=30, n_init=3)
            print("✅ DeepSORT inicializado")
            
        except Exception as e:
            print(f"❌ Error inicializando modelos: {e}")
            # En un entorno de servidor como HF Spaces, es mejor lanzar el error
            # para que el espacio se reinicie o muestre un estado de error.
            raise
    
    def reset_stats(self):
        """Reinicia las estadísticas de rendimiento."""
        self.stats = {
            'fps': 0,
            'frame_count': 0,
            'object_count': 0,
            'faces_detected': 0,
            'processing_time': 0
        }
    
    def detect_objects(self, frame: np.ndarray) -> List[dict]:
        """Detecta objetos usando YOLO."""
        # El argumento 'verbose=False' evita que YOLO imprima logs en la consola
        results = self.yolo_model(frame, verbose=False)
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    
                    if conf > 0.5:
                        detections.append({
                            'bbox': [int(x1), int(y1), int(x2-x1), int(y2-y1)],
                            'confidence': float(conf),
                            'class': cls,
                            'class_name': self.yolo_model.names[cls]
                        })
        
        return detections
    
    def apply_blur(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], 
                   blur_intensity: int = 25) -> np.ndarray:
        """Aplica desenfoque gaussiano a una región específica."""
        x, y, w, h = bbox
        
        # Asegurar que la región esté dentro de los límites del frame
        x, y = max(0, x), max(0, y)
        w, h = min(w, frame.shape[1] - x), min(h, frame.shape[0] - y)
        
        if w > 0 and h > 0:
            roi = frame[y:y+h, x:x+w]
            # La intensidad del desenfoque debe ser un número impar
            kernel_size = (blur_intensity // 2) * 2 + 1
            blurred_roi = cv2.GaussianBlur(roi, (kernel_size, kernel_size), 0)
            frame[y:y+h, x:x+w] = blurred_roi
        
        return frame
    
    def process_video(self, input_path: str, blur_people: bool, 
                     blur_intensity: int, progress=gr.Progress()) -> Tuple[str, str]:
        """Procesa el video completo aplicando desenfoque."""
        if input_path is None:
            return None, "❌ Error: Por favor, sube un video primero."

        try:
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                return None, "❌ Error: No se pudo abrir el archivo de video."
            
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            output_path = "output_blurred.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            self.reset_stats()
            start_time = time.time()
            
            for frame_idx in progress.tqdm(range(total_frames), desc="Procesando video..."):
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Detectar objetos (solo si es necesario desenfocar)
                if blur_people:
                    detections = self.detect_objects(frame)
                    
                    # Preparar detecciones para DeepSORT (solo de personas)
                    detection_list = []
                    for det in detections:
                        if det['class_name'] == 'person':
                            bbox = det['bbox']
                            conf = det['confidence']
                            detection_list.append(([bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]], conf, 'person'))
                    
                    # Actualizar tracker y aplicar desenfoque
                    tracks = self.tracker.update_tracks(detection_list, frame=frame)
                    for track in tracks:
                        if not track.is_confirmed():
                            continue
                        
                        bbox = track.to_ltrb()
                        x1, y1, x2, y2 = [int(x) for x in bbox]
                        frame = self.apply_blur(frame, (x1, y1, x2 - x1, y2 - y1), blur_intensity)
                
                out.write(frame)
            
            cap.release()
            out.release()
            
            total_time = time.time() - start_time
            avg_fps = total_frames / total_time if total_time > 0 else 0
            
            stats_report = f"""
            📊 Estadísticas de Procesamiento:
            • Frames procesados: {total_frames}
            • Tiempo total: {total_time:.2f}s
            • FPS promedio: {avg_fps:.2f}
            • Dispositivo usado: {self.device}
            """
            
            return output_path, f"✅ ¡Video procesado exitosamente!\n{stats_report}"
            
        except Exception as e:
            # Proporcionar un mensaje de error más detallado
            import traceback
            traceback.print_exc()
            return None, f"❌ Error procesando el video: {str(e)}"
    
    def create_interface(self):
        """Crea la interfaz de Gradio."""
        with gr.Blocks(title="🎭 Desenfoque Automático de Caras", theme=gr.themes.Soft()) as interface:
            gr.Markdown("""
            # 🎭 Aplicación de Desenfoque Automático de Caras
            Esta aplicación utiliza **YOLOv8** y **DeepSORT** para detectar, seguir y desenfocar automáticamente 
            personas en videos para proteger la privacidad.
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 📤 Entrada y Configuración")
                    input_video = gr.File(
                        label="Subir Video",
                        file_types=[".mp4", ".avi", ".mov"],
                        type="filepath" # Gradio maneja la ruta del archivo temporal
                    )
                    
                    blur_people = gr.Checkbox(
                        label="Desenfocar Personas",
                        value=True,
                        info="Detecta y desenfoca personas automáticamente."
                    )
                    
                    blur_intensity = gr.Slider(
                        minimum=5, maximum=99, value=35, step=2,
                        label="Intensidad de Desenfoque",
                        info="Valor más alto = más desenfoque (debe ser impar)."
                    )
                    
                    process_btn = gr.Button("🚀 Procesar Video", variant="primary")
                
                with gr.Column(scale=1):
                    gr.Markdown("### 📥 Resultado")
                    output_video = gr.Video(label="Video Procesado", height=400)
                    status_text = gr.Textbox(
                        label="Estado del Procesamiento",
                        lines=8,
                        interactive=False,
                        placeholder="El estado y las estadísticas aparecerán aquí..."
                    )
            
            process_btn.click(
                fn=self.process_video,
                inputs=[input_video, blur_people, blur_intensity],
                outputs=[output_video, status_text]
            )

            gr.Examples(
                examples=[
                    ["examples/people.mp4", True, 35],
                    ["examples/street.mp4", True, 51],
                ],
                inputs=[input_video, blur_people, blur_intensity],
                outputs=[output_video, status_text],
                fn=self.process_video,
                cache_examples=True # Acelera la carga de ejemplos
            )

            gr.Markdown("--- \n *Hecho con Gradio, YOLOv8 y DeepSORT. Adaptado para Hugging Face Spaces.*")
        
        return interface

# --- Función Principal ---
if __name__ == "__main__":
    print("🚀 Iniciando Aplicación de Desenfoque Automático...")
    app_instance = FaceBlurApp()
    interface = app_instance.create_interface()
    
    # En HF Spaces, no necesitas 'share=True' ni configurar server_name/port.
    # 'debug=True' es útil para ver los errores detallados en los logs del Space.
    interface.launch(debug=True)