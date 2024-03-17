import cv2
import easyocr

reader = easyocr.Reader(['fr']) # specify the language

while True:
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()

    # Vérifier si la lecture de l'image est réussie
    if not ret:
        print("Impossible de capturer l'image.")
        break

    # Afficher l'image capturée
    # cv2.imshow('Camera Feed', frame)

    # Attendre 1 milliseconde entre chaque image et vérifier si une touche est pressée
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Convertir l'image en niveaux de gris
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    result = reader.readtext(image)

    print('-------------------------------------------------')
    t = []
    for (bbox, text, prob) in result:
        if prob > 0.8:
            t.append(f'Text: {text}, Probability: {prob}')
            break

    print(t)

cv2.destroyAllWindows()
