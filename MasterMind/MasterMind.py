import random
import cv2
import numpy as np
import easyocr
import requests

N = 8
P = 4


class OCR:
    """
    Interface pour la reconnaissance de caractères.
    Première implémentation avec la caméra intégrée au PC
    """
    def __init__(self):
        self.reader = easyocr.Reader(['fr'])  # Utiliser EasyOCR avec la langue anglaise
        self.width = 640
        self.height = 480

    def shape(self):
        return self.height, self.width

    def internal_camera(self):
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        return frame

    def esp32cam(self):
        r = requests.get("http://192.168.4.1:80/capture")
        image = np.asarray(bytearray(r.content), dtype=np.uint8)
        return cv2.imdecode(image, cv2.IMREAD_COLOR)

    def read(self):
        frame = self.internal_camera()
        # frame = self.esp32cam()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        return self.reader.readtext(frame), frame


class Jeu:
    info_start = "Choisis une position"

    def __init__(self):
        self.info = Jeu.info_start
        self.jeu = [-1 for i in range(P)]
        self.position = -1


class MastermindCV:
    def __init__(self):
        self.ocr = OCR()
        self.frame = None

        # Créer une fenêtre OpenCV
        cv2.namedWindow('MasterMind')

        # initialise les lignes d'info
        self.info_start = "Choisis une position"
        self.info = []
        self.info.append(self.info_start)

        # Définir les valeurs possibles
        self.valeurs = [i for i in range(1, N + 1)]
        
        # initialise la combinaison secrète
        self.secret = random.sample(self.valeurs, P)

        self.jeux = []
        self.jeux.append(Jeu())

        # initialise les jeux successives
        self.lignes = 1

        # print("valeurs", self.valeurs, "code", self.secret)

        # Dessiner l'IHM
        self.draw_ihm()

    def jeu_courant(self):
        return self.jeux[self.lignes - 1]

    def result(self):
        """
        analyse la combinaison choisie 
        """
        exact = 0
        exists = 0
        off = 0
        jeu = self.jeu_courant()
        for p in range(P):
            k = jeu.jeu[p]
            if k == self.secret[p]:
                exact += 1
            elif k in self.secret:
                exists += 1
            else:
                off += 1

        r = False
        if exact == P:
            jeu.info = f"Bravo !!!"
            r = True
        else:
            jeu.info = f"OK={exact} on={exists} off={off}"
        # print("result", jeu.info)

        self.draw_ihm()

        return r

    # Fonction pour dessiner l'IHM
    def draw_ihm(self, current_position=-1):
        # print("draw_ihm. Position=", current_position, "lignes=", self.lignes, "info=", self.info)

        position_width = 70
        position_height = 50
        padding = 10

        # zone d'information
        info_height = int(position_height*0.6)

        # chaque ligne de jeu
        ligne_height = padding + position_height + padding + info_height

        # l'image complète de l'IHM
        self.width1 = padding + P * (position_width + padding)
        self.height1 = padding + self.lignes * ligne_height

        width = self.width1 + self.ocr.width
        height = self.ocr.height
        if self.height1 > 480: height = self.height1

        self.image = np.zeros((height, width, 3), dtype=np.uint8)

        # np.copy()

        y = 0

        # on affiche successivement toutes les tentatives de combinaisons
        for ligne, jeu in enumerate(self.jeux):

            x1 = padding
            y1 = y + padding
            labels = ['A', 'B', 'C', 'D']
            for position in range(P):
                x2 = x1 + position_width
                y2 = y1 + position_height
                # Dessiner les zones sur l'image 
                # la couleur change pour la zone en cours 
                c = (255, 0, 0)
                if ligne == (self.lignes - 1) and position == current_position:
                    c = (0, 255, 0)

                # print("draw_ihm. i=", i, x1, y1, x2, y2)
                cv2.rectangle(self.image, (x1, y1), (x2, y2), c, -1)

                cv2.putText(self.image, labels[position], (x1 + 10, y1 + int(position_height*0.3)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                            cv2.LINE_AA)

                j = jeu.jeu[position]
                # print("draw_ihm. position=", position, i, "jeu=", j)
                if j > 0:
                    cv2.putText(self.image, f"{j}", (x1 + 30, y1 + int(position_height*0.8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 1,
                                cv2.LINE_AA)

                x1 += position_width + padding

            x1 = padding
            y1 = y + padding + position_height + padding

            x2 = x1 + P * (position_width + padding) - padding
            y2 = y1 + info_height
            c = (255, 255, 255)

            cv2.rectangle(self.image, (x1, y1), (x2, y2), c, -1)  # Carré 1 (bleu)
            cv2.putText(self.image, jeu.info, (x1 + 10, y1 + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1,
                        cv2.LINE_AA)

            y += ligne_height

        if self.frame is not None:
            # print(self.frame.shape)
            self.image[0:self.ocr.height,self.width1:self.width1 + self.ocr.width] = self.frame

        # Afficher l'image
        cv2.imshow('MasterMind', self.image)

    def process_frame(self):
        def contains_integer(text):
            try:
                int(text)
                return True
            except ValueError:
                return False

        result, self.frame = self.ocr.read()

        jeu = self.jeu_courant()

        for (bbox, text, prob) in result:
            if prob > 0.8 and contains_integer(text):
                t = int(text)
                if t > 0 and t <= 8:
                    # print("t=", t, "position=", jeu.position, "jeu=", jeu)
                    if jeu.position >= 0:
                        jeu.jeu[jeu.position] = t
                        # print("process_frame. position=", self.position, "jeu=", jeu)
                        self.draw_ihm(jeu.position)

                    break

    def run(self):
        while True:
            # Traitement de l'image pour détecter les chiffres et les reconnaître
            self.process_frame()

            # détection des touches du clavier
            k = cv2.waitKey(1) & 0xFF
            if k != 255:
                # print("k=", k)
                pass
            if k == ord('q'):
                # quit
                break

            zone = -1
            if k == ord('a') or k == ord('A'):
                zone = 0
            elif k == ord('b') or k == ord('B'):
                zone = 1
            elif k == ord('c') or k == ord('C'):
                zone = 2
            elif k == ord('d') or k == ord('D'):
                zone = 3
            elif k == 13:
                # enter => valider une combinaison
                zone = 4

            jeu = self.jeu_courant()

            if zone >= 0 and zone <= 3:
                # print("zone=", zone)
                self.draw_ihm(zone)
                jeu.position = zone
            if zone == 4:
                # on teste la combinaison
                ok = self.result()
                if not ok:
                    self.lignes += 1
                    self.jeux.append(Jeu())
                    self.draw_ihm()

        cv2.destroyAllWindows()


# Fonction principale
def main():
    cv_game = MastermindCV()
    cv_game.run()


if __name__ == "__main__":
    main()
