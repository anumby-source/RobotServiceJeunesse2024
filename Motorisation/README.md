

https://www.aranacorp.com/fr/utilisation-dun-module-l298n-avec-arduino/

# comment puis-je réguler la vitesse relative des deux moteurs?

Pour réguler la vitesse relative des deux moteurs à l'aide du L298N avec un Arduino Uno, vous utiliserez la modulation de largeur d'impulsion (PWM, pour Pulse Width Modulation en anglais).

Voici comment vous pouvez le faire:

Connexions:

* Connectez les broches OUT1 et OUT2 du L298N au premier moteur.
* Connectez les broches OUT3 et OUT4 au deuxième moteur.
* Les broches IN1 et IN2 contrôlent la direction du premier moteur.
* Les broches IN3 et IN4 contrôlent la direction du deuxième moteur.
* Les broches ENA et ENB du L298N contrôlent respectivement la vitesse des moteurs 1 et 2 grâce à la modulation de largeur d'impulsion (PWM). Connectez-les aux broches PWM de votre Arduino (par exemple, les broches 9 et 10).

##  Code:
Vous pouvez utiliser le code suivant pour contrôler les moteurs avec votre Arduino:

```
// Définition des broches
const int IN1 = 2;
const int IN2 = 3;
const int ENA = 9; // broche PWM

const int IN3 = 4;
const int IN4 = 5;
const int ENB = 10; // broche PWM

void setup() {
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(ENA, OUTPUT);

  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);
  pinMode(ENB, OUTPUT);
}

void loop() {
  // Pour faire tourner le moteur 1 dans une direction à pleine vitesse
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  analogWrite(ENA, 255); // 255 est la vitesse maximale

  // Pour faire tourner le moteur 2 dans une direction à demi-vitesse
  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);
  analogWrite(ENB, 127); // 127 est la moitié de la vitesse maximale

  delay(2000); // attend 2 secondes

  // Pour arrêter les moteurs
  analogWrite(ENA, 0);
  analogWrite(ENB, 0);

  delay(2000); // attend 2 secondes
}
```

(bien sûr ChatGPT ne peut pas connecter les connections réelles que l'on utilise)

Pour réguler la vitesse relative des deux moteurs, ajustez simplement les valeurs que vous fournissez à la fonction analogWrite() pour les broches ENA et ENB. La valeur peut être comprise entre 0 (arrêt total) et 255 (vitesse maximale).

# Je voudrais piloter le circuit L298N avec un ESP8266 pour piloter 2 moteurs continu quel sera le cablage?

Piloter un L298N avec un ESP8266 est une bonne idée pour ajouter des capacités sans fil à votre projet de robotique. Voici comment vous pouvez câbler le L298N avec un ESP8266 pour piloter deux moteurs à courant continu:

##  Câblage:
Alimentation:

* GND du L298N à la broche GND de l'ESP8266.
* VCC ou +12V (ou l'entrée d'alimentation pour les moteurs) du L298N à votre source d'alimentation pour les moteurs (par exemple, une batterie 12V).
* 5V ou +5V du L298N (c'est l'entrée d'alimentation pour la logique) peut être reliée à une source d'alimentation externe 5V. Ne le connectez pas directement à l'ESP8266 car ce dernier fonctionne à 3.3V et peut être endommagé par une tension de 5V.

## Moteurs:
* OUT1 et OUT2 du L298N au premier moteur.
* OUT3 et OUT4 au deuxième moteur.

## Commandes:

* Les broches IN1 et IN2 du L298N peuvent être connectées à n'importe quelle GPIO de l'ESP8266, par exemple GPIO4 et GPIO5.
* Les broches IN3 et IN4 du L298N peuvent également être connectées à n'importe quelle autre GPIO, par exemple GPIO12 et GPIO13.
* Les broches ENA et ENB du L298N doivent être connectées aux broches GPIO qui supportent le PWM de l'ESP8266, par exemple GPIO14 (D5) et GPIO16 (D0).
* Note: Le L298N a parfois des diodes intégrées pour la protection contre les courants inverses, mais si le vôtre n'en a pas, vous voudrez peut-être ajouter des diodes de roue libre (flyback diodes) pour protéger le driver contre les courants inverses générés par les moteurs.

Autre point important: Assurez-vous que les niveaux de tension sont compatibles. L'ESP8266 fonctionne à 3,3V. Si le L298N ne reconnaît pas correctement 3,3V comme une entrée "HIGH", vous pourriez avoir besoin d'un circuit de niveau de tension (level shifter) entre eux. Cependant, dans de nombreux cas, le L298N reconnaît 3,3V comme "HIGH" sans problèmes.

Après avoir établi le câblage, vous pouvez utiliser l'IDE Arduino avec la bibliothèque ESP8266 pour programmer votre ESP8266 et contrôler les moteurs. Utilisez la fonction analogWrite pour envoyer une valeur PWM aux broches ENA et ENB pour réguler la vitesse des moteurs.

# Configuration ESP8266 + un circuit LN298N et 2 moteurs à courant continu. Je veux ajouter une télécommande infrarouge pour commander les moteurs

Pour ajouter une télécommande infrarouge (IR) pour commander les moteurs avec votre configuration ESP8266 et L298N, vous aurez besoin d'un récepteur IR (par exemple, un module IR comme le TSOP38238) et d'une télécommande IR compatible. Voici comment vous pouvez procéder :

## Connexion du récepteur IR :
* Connectez la broche de signal du récepteur IR à une broche d'entrée numérique de votre ESP8266.
* Connectez la broche VCC du récepteur IR à une source d'alimentation 3,3V de l'ESP8266.
* Connectez la broche GND du récepteur IR à la masse (GND) de l'ESP8266.

## Programmation IR :
Vous devrez utiliser une bibliothèque IR pour Arduino compatible avec votre ESP8266. Une bibliothèque couramment utilisée est "IRremoteESP8266". Vous pouvez l'installer depuis le gestionnaire de bibliothèques Arduino.
Utilisez la bibliothèque pour écouter les signaux infrarouges à partir de la télécommande.

## Associations télécommande-moteur :
Associez chaque bouton de la télécommande à une action spécifique sur les moteurs, comme l'activation/désactivation, la régulation de la vitesse, ou la modification de la direction.
Dans votre code, déterminez quelle action doit être effectuée en fonction du code infrarouge reçu.

## Contrôle des moteurs :
Utilisez les broches de commande de votre L298N connectées à votre ESP8266 pour contrôler les moteurs en fonction des commandes infrarouges que vous avez associées.
Par exemple, lorsque vous recevez une commande infrarouge spécifique, vous pouvez activer/désactiver les broches IN1 et IN2 pour contrôler la direction du moteur 1, et les broches IN3 et IN4 pour contrôler le moteur 2.
Voici un exemple de code pour vous aider à démarrer avec la bibliothèque IRremoteESP8266 :

```
#include <IRremoteESP8266.h>
#include <IRrecv.h>
#include <IRutils.h>

const int receiverPin = D2; // Broche du récepteur IR
IRrecv irrecv(receiverPin);
decode_results results;

void setup() {
  irrecv.enableIRIn(); // Initialise le récepteur IR
  // Initialisez ici la configuration de votre L298N et de vos moteurs
}

void loop() {
  if (irrecv.decode(&results)) {
    // Traitez ici les commandes infrarouges reçues
    // Par exemple, utilisez un switch/case pour associer les commandes aux actions des moteurs
    switch (results.value) {
      case 0xFFA25D: // Remplacez ce code par le code de la commande que vous voulez gérer
        // Exécutez une action sur les moteurs en réponse à cette commande
        break;
      // Ajoutez d'autres cas pour d'autres commandes
    }
    irrecv.resume(); // Réactive le récepteur pour la prochaine commande
  }
}
```

N'oubliez pas de remplacer les codes de commande IR par ceux de votre télécommande. Vous pouvez utiliser la fonction irrecv.decode(&results) pour lire les codes IR reçus et les associer à des actions sur vos moteurs.


# Je pilote deux moteurs courant continu avec un circuit L298N. Je voudrais mesurer la vitesse de rotation des moteurs.

Pour mesurer la vitesse de rotation des moteurs à courant continu, l'une des méthodes les plus courantes consiste à utiliser des encodeurs. Les encodeurs sont des dispositifs qui transforment le mouvement rotatif ou la position en un signal électrique que vous pouvez lire.

Il existe deux types principaux d'encodeurs: les encodeurs incrémentaux et les encodeurs absolus. Pour mesurer la vitesse, un encodeur incrémental est généralement suffisant. Cet encodeur génère un certain nombre d'impulsions par tour de l'arbre du moteur, permettant de déterminer à la fois la vitesse et la direction de rotation.

Voici les étapes pour mesurer la vitesse de rotation des moteurs à l'aide d'un encodeur incrémental:

## Installation de l'encodeur:

Achetez un encodeur adapté à votre moteur. Il pourrait s'agir d'un module encodeur attaché à l'arbre du moteur ou d'un encodeur optique qui lit les marques sur un disque monté sur l'arbre du moteur.
Montez l'encodeur conformément aux instructions du fabricant.

### Câblage:
Connectez la sortie de l'encodeur (souvent appelée canal A ou sortie d'impulsion) à une entrée numérique de votre microcontrôleur (Arduino, ESP8266, etc.).
Connectez également les broches d'alimentation (VCC et GND) de l'encodeur aux broches appropriées de votre microcontrôleur ou à une source d'alimentation adaptée.
### Programmation:
Configurez l'interruption sur l'entrée numérique pour détecter chaque impulsion provenant de l'encodeur. Chaque fois qu'une impulsion est détectée, cela signifie que le moteur a tourné d'un certain angle.
Pour mesurer la vitesse, comptez le nombre d'impulsions sur une période donnée (par exemple, une seconde). La vitesse sera proportionnelle au nombre d'impulsions.

Exemple de code pour Arduino:

```
const int encoderPin = 2; // La broche où est connectée la sortie de l'encodeur
volatile int pulseCount = 0; // Nombre d'impulsions détectées

void setup() {
  Serial.begin(9600);
  pinMode(encoderPin, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(encoderPin), countPulse, RISING); // Interruption sur front montant
}

void loop() {
  delay(1000); // Attend 1 seconde
  detachInterrupt(digitalPinToInterrupt(encoderPin)); // Désactive l'interruption
  int pulsesPerSecond = pulseCount; // Lit le nombre d'impulsions sur la dernière seconde
  pulseCount = 0; // Réinitialise le compteur
  attachInterrupt(digitalPinToInterrupt(encoderPin), countPulse, RISING); // Réactive l'interruption
  Serial.println(pulsesPerSecond); // Affiche le nombre d'impulsions par seconde
}

void countPulse() {
  pulseCount++;
}
```

Note: Vous devrez connaître le nombre d'impulsions générées par l'encodeur par tour complet du moteur pour convertir le nombre d'impulsions par seconde en rotations par minute (RPM).

# comment utiliser le module HC020K pour mesurer la vitesse de rotation ?

Le module HC020K est un module d'encodeur à effet Hall couramment utilisé pour mesurer la vitesse de rotation des moteurs à courant continu. Il comprend généralement deux composants principaux: l'encodeur lui-même (qui détecte le passage d'aimants à l'aide d'un capteur à effet Hall) et un disque à aimants (ou des aimants intégrés à la roue du moteur).

Pour mesurer la vitesse de rotation avec le module HC020K, suivez les étapes suivantes:

## Installation:
Fixez le disque à aimants sur l'arbre du moteur ou assurez-vous que les aimants sont bien fixés sur la roue du moteur.
Placez l'encodeur HC020K de manière à ce que le capteur à effet Hall soit à proximité des aimants lorsqu'ils tournent.
## Câblage:
Connectez la broche VCC du HC020K à une source d'alimentation de 5V (si vous utilisez un Arduino ou une autre plateforme compatible 5V).
Connectez la broche GND au GND de votre plateforme.
Connectez la broche OUT (sortie de signal) à une broche d'entrée numérique de votre plateforme.
## Programmation:
Configurez une interruption sur la broche d'entrée pour détecter chaque passage d'aimant (c'est-à-dire chaque impulsion). La vitesse de rotation sera proportionnelle au nombre d'impulsions détectées pendant un intervalle de temps donné.

Exemple de code pour Arduino:

```
const int encoderPin = 2; // La broche où est connectée la sortie OUT du HC020K
volatile int pulseCount = 0; // Nombre d'impulsions détectées

void setup() {
  Serial.begin(9600);
  pinMode(encoderPin, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(encoderPin), countPulse, RISING); // Interruption sur front montant
}

void loop() {
  delay(1000); // Attend 1 seconde
  detachInterrupt(digitalPinToInterrupt(encoderPin)); // Désactive l'interruption
  int pulsesPerSecond = pulseCount; // Lit le nombre d'impulsions sur la dernière seconde
  pulseCount = 0; // Réinitialise le compteur
  attachInterrupt(digitalPinToInterrupt(encoderPin), countPulse, RISING); // Réactive l'interruption
  Serial.println(pulsesPerSecond); // Affiche le nombre d'impulsions par seconde
}

void countPulse() {
  pulseCount++;
}
```

Note: Pour convertir le nombre d'impulsions par seconde en rotations par minute (RPM), vous devez connaître le nombre d'aimants (ou de pôles magnétiques) sur votre disque ou roue. Par exemple, si vous avez 20 aimants, et que vous comptez 60 impulsions en une seconde, cela signifie que le moteur a fait 3 tours complets en cette seconde (60/20 = 3 tours par seconde, soit 180 tours par minute).

https://arduino.blaisepascal.fr/les-codeurs-incrementaux/

# Et en utilisant un encodeur optique ?

L'utilisation d'un encodeur optique est une autre excellente méthode pour mesurer la vitesse de rotation d'un moteur. Un encodeur optique fonctionne en détectant les passages de fentes ou de marques sur un disque à travers lequel une lumière LED est émise et captée par un phototransistor.

Voici les étapes pour mesurer la vitesse de rotation avec un encodeur optique:

Installation:

Fixez le disque avec des fentes (généralement appelé disque encodeur) sur l'arbre du moteur.
Montez l'encodeur optique de manière à ce qu'il puisse lire le passage des fentes du disque lorsqu'il tourne.
Câblage:

Connectez la broche VCC ou + de l'encodeur à une source d'alimentation (généralement 5V).
Connectez la broche GND ou - au GND de votre plateforme (par exemple, Arduino).
Connectez la broche de sortie de l'encodeur (généralement désignée OUT ou SIGNAL) à une broche d'entrée numérique de votre plateforme.
Programmation:

Comme avec l'encodeur à effet Hall, configurez une interruption sur la broche d'entrée pour détecter chaque passage de fente. La vitesse de rotation sera proportionnelle au nombre de fentes détectées pendant un intervalle de temps donné.
Exemple de code pour Arduino:

```
const int encoderPin = 2; // La broche où est connectée la sortie de l'encodeur optique
volatile int pulseCount = 0; // Nombre de fentes détectées

void setup() {
  Serial.begin(9600);
  pinMode(encoderPin, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(encoderPin), countPulse, RISING); // Interruption sur front montant
}

void loop() {
  delay(1000); // Attend 1 seconde
  detachInterrupt(digitalPinToInterrupt(encoderPin)); // Désactive l'interruption
  int pulsesPerSecond = pulseCount; // Lit le nombre de fentes détectées sur la dernière seconde
  pulseCount = 0; // Réinitialise le compteur
  attachInterrupt(digitalPinToInterrupt(encoderPin), countPulse, RISING); // Réactive l'interruption
  Serial.println(pulsesPerSecond); // Affiche le nombre de fentes détectées par seconde
}

void countPulse() {
  pulseCount++;
}
```

Note: Pour convertir le nombre de fentes détectées par seconde en RPM, vous devez connaître le nombre total de fentes sur le disque encodeur. Par exemple, si le disque a 20 fentes et que vous comptez 60 fentes en une seconde, cela signifie que le moteur a fait 3 tours complets en cette seconde (60/20 = 3 tours par seconde, soit 180 RPM).

