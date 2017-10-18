import numpy as np
import time
import cv2

device = 1
cam = cv2.VideoCapture(device)

if not cam.isOpened():
    cam.open(device)
    
if cam.isOpened:
    
    # cargamos las etiquetas de clase desde el disco
    rows = open('synset_words.txt').read().strip().split("\n")
    classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]
    
    # cargar nuestro modelo serializado desde el disco
    print("[INFO] cargando modelo...")
    net = cv2.dnn.readNetFromCaffe('synset_words.txt', 'bvlc_googlenet.caffemodel')
    
    while(True):
        ret, frame = cam.read()
        
        '''
        cv2.rectangle(frame, (300, 300), (100, 100), (0, 255, 0), 0)
        crop_img = frame[100:300, 100:300]
        grey = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        value = (37, 37)
        blurred = cv2.GaussianBlur(grey, value, 0)
        _, thresh1 = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        _, contours, hierarchy = cv2.findContours(thresh1.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        
        max_area = -1
        print len(contours)
        for i in range(len(contours)):
            cnt = contours[i]
            area = cv2.contourArea(cnt)
            if (area > max_area):
                max_area = area
                ci = i
        cnt = contours[ci]
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(crop_img, (x, y), (x + w, y + h), (0, 0, 255), 0)
        hull = cv2.convexHull(cnt)
        drawing = np.zeros(crop_img.shape, np.uint8)
        cv2.drawContours(drawing, [cnt], 0, (0, 255, 0), 0)
        cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 0)
        hull = cv2.convexHull(cnt, returnPoints=False)
        defects = cv2.convexityDefects(cnt, hull)
        count_defects = 0
        cv2.drawContours(thresh1, contours, -1, (0, 255, 0), 3)
        
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(cnt[s][0])
            end = tuple(cnt[e][0])
            far = tuple(cnt[f][0])
            a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57
            if angle <= 90:
                count_defects += 1
                cv2.circle(crop_img, far, 1, [0, 0, 255], -1)
            dist = cv2.pointPolygonTest(cnt, far, True)
            cv2.line(crop_img, start, end, [0, 255, 0], 2)
            cv2.circle(crop_img, far, 5, [0, 0, 255], -1)
            
        drawing = cv2.flip(drawing, 1) # Cambio la imagen a modo espejo
        crop_img = cv2.flip(crop_img, 1)
        frame = cv2.flip(frame, 1)
    
        if count_defects > 3:
            cv2.putText(frame, "ABIERTA", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 4)
        else:
            cv2.putText(frame, "CERRADA", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 4)
    
        cv2.imshow('drawing', drawing)
        cv2.imshow('end', crop_img)
        cv2.imshow('Gesture', frame)
        
        all_img = np.hstack((drawing, crop_img))
        cv2.imshow('Contours', all_img)
        
        '''
        
        frame = cv2.flip(frame, 1)
        cv2.imshow('video', frame)        
        
        
        
        
        
        # nuestra CNN requiere dimensiones espaciales fijas para nuestra(s) imagen(es) de entrada
        # por lo que debemos asegurarnos de que se redimensione a 224x224 pixeles mientras se
        # realiza una resta de media (104, 117, 123) para normalizar la entrada;
        # despues de ejecutar este comando nuestro "blob" ahora tiene la forma(shape):
        # (1, 3, 224, 224)
        
        blob = cv2.dnn.blobFromImage(frame, 1, (224, 224), (104, 117, 123))
        
        '''
        
        # establecer el blob como entrada a la red y realizar un
        # forward-pass para obtener nuestra clasificacion de salida
        net.setInput(blob)
        start = time.time()
        preds = net.forward()
        end = time.time()
        print("[INFO] la classification ha tomado {:.5} segundos".format(end - start))
        '''
        # clasificamos los indices de las probabilidades en orden descendente
        # (las probabilidades mas alta primero) y agarramos las predicciones de las 5 mejores
        '''
        idxs = np.argsort(preds[0])[::-1][:5]
        
        # bucle sobre las predicciones de las 5 mejores y mostrarlas
        for (i, idx) in enumerate(idxs):
            # dibujar la prediccion superior en la imagen de entrada
            if i == 0:
                text = "Etiqueta: {}, {:.2f}%".format(classes[idx],
                                                   preds[0][idx] * 100)
                cv2.putText(frame, text, (5, 25), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 255), 2)
        
            # mostramos la etiqueta predicha + la probabilidad asociada a la consola
            print("[INFO] {}. etiqueta: {}, probabilidad: {:.5}".format(i + 1, classes[idx], preds[0][idx]))
        
        
            '''
        
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cam.release()
    cv2.destroyAllWindows()
else:
    print "Ocurrio un problema al intentar abrir el dispositivo: " + device