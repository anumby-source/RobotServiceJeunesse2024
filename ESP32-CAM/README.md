
# configuration de L'IDE Arduino pour compiler et lancer le code de la caméra

Vérifier la version du "board système" de l'ESP32

1) sélectionner "esp32" dans les boards manager
2) vérifier la version du board "esp32 by expressif"
3) si la version 2.0.xxx est installée ===>  suprimer cette version

4) installer la version 1.0.6

ESP32-CAM\Conf ESP32 board IDE .png
Puis vérifier que l'on sélectionne "AI Thinker ESP32-CAM"

Et ça doit fonctionner

Rappel:

1) connecter le réseau

const char* ssid = "ESP32-CAM Access Point";
const char* password = "123456789";

2) ouvrir un navigateur à l'adresse:

192.168.4.1

ESP32-CAM/Control CAM.png

