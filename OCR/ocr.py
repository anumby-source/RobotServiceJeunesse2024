import cv2
import pytesseract
from pytesseract import Output
import numpy as np

while True:
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cv2.imwrite("image.jpg", frame)
    cap.release()

    # Charger et adapter l'image capturée
    image = cv2.imread("image.jpg")

    # Convertir l'image en niveaux de gris
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Appliquer un flou pour réduire le bruit
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Détection des contours dans l'image
    edged = cv2.Canny(blurred, 30, 150)

    # Trouver les contours dans l'image
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialiser la liste des régions d'intérêt
    ROIs = []

    # Parcourir tous les contours trouvés
    for contour in contours:
        # Ignorer les petits contours
        if cv2.contourArea(contour) > 100:
            x, y, w, h = cv2.boundingRect(contour)
            ROI = gray[y:y+h, x:x+w]
            ROIs.append((x, y, w, h, ROI))

    # Initialiser une liste pour stocker les chiffres détectés
    detected_numbers = []

    pytesseract.pytesseract.tesseract_cmd = r'chemin_vers_tesseract'
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    numbers = []
    # Utiliser Tesseract OCR pour reconnaître les chiffres dans les régions d'intérêt
    for (x, y, w, h, ROI) in ROIs:
        custom_config = r'--oem 3 --psm 6 outputbase digits'
        number = pytesseract.image_to_string(ROI, config=custom_config)
        numbers.append(number)
        detected_numbers.append((x, y, number))

    print(numbers)

    """

    d = pytesseract.image_to_data(image, output_type=Output.DICT)
    NbBoites = len(d['level'])
    print("Nombre de boites: " + str(NbBoites))
    for i in range(NbBoites):
        # Récupère les coordonnées de chaque boite
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        # Affiche un rectangle
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    """

    # Afficher les chiffres détectés
    for (x, y, number) in detected_numbers:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, number, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


    # Afficher l'image avec les chiffres détectés
    cv2.imshow('img', image)
    cv2.waitKey(0)

cv2.destroyAllWindows()



try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract
from pytesseract import Output
import cv2
simage = r'/[Path to image...]/image_2.png'
img = cv2.imread(simage)
cv2.destroyAllWindows()