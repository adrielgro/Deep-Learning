from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2

# constructor
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--confianza", type=float, default=0.2,
	help="probabilidad minima para filtrar detecciones debiles")
args = vars(ap.parse_args())

# inicializar la lista de clases etiquetas de la clase MobileNet SSD
# que fue entrenada para detectar, luego genera un conjunto de colores 
# del cuadro delimitador para cada clase
CLASSES = ["fondo", "avion", "bicicleta", "pajaro", "barco",
	"botella", "autobus", "carro", "gato", "silla", "vaca", "comedor",
	"perro", "caballo", "motorbike", "persona", "maceta", "oveja",
	"sofa", "tren", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# cargar el modelo serializado
print("[INFO] cargando modelo...")
net = cv2.dnn.readNetFromCaffe('test.prototxt.txt', 'test.caffemodel')

# inicializamos el streaming de video, y le permitimos al sensor de la
# camara que se prepare e incialize el contador FPS
print("[INFO] iniciando streaming de video...")
vs = VideoStream(src=1).start()
time.sleep(2.0)
fps = FPS().start()

# bucle sobre los fotogramas de la transmision de video
while True:
	# toma el marco de la secuencia de video subproceso y cambia su tamano 
    # para que tenga un ancho maximo de 400 píxeles
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	# toma las dimensiones del marco y lo convierte en un blob
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
		0.007843, (300, 300), 127.5)

	# pasa el blob a traves de la red y obtiene las detecciones 
    # y predicciones
	net.setInput(blob)
	detections = net.forward()

	# bucle sobre las detecciones
	for i in np.arange(0, detections.shape[2]):
		# Extraer la confianza (es decir, la probabilidad) asociada 
         # con la predicción
		confidence = detections[0, 0, i, 2]

		# filtra las detecciones debiles asegurando que la "confianza" 
         # sea mayor que la confianza minima
		if confidence > args["confianza"]:
			# extract the index of the class label from the
			# `detections`, then compute the (x, y)-coordinates of
			# the bounding box for the object
			idx = int(detections[0, 0, i, 1])
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# dibuja la prediccion en el frame
			label = "{}: {:.2f}%".format(CLASSES[idx],
				confidence * 100)
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				COLORS[idx], 2)
			y = startY - 15 if startY - 15 > 15 else startY + 15
			cv2.putText(frame, label, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

	# mostramos los frames de salida
	cv2.imshow("Camara UABC", frame)
	key = cv2.waitKey(1) & 0xFF

	# presionar la letra q para salir del ciclo
	if key == ord("q"):
		break

	# actualizar el contador fps
	fps.update()

# detener el temporizador y mostrar la informacion FPS
fps.stop()
print("[INFO] tiempo transcurrido: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()