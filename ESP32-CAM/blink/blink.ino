// Inclure la bibliothèque ESP8266
#include <ESP8266WiFi.h>


// Broche à laquelle est connectée la LED (D0/GPIO 16 sur NodeMCU)
const int ledPin = 2; // D4   GPIO2    TXD1

void setup() {
  // Initialisation de la broche de la LED en sortie
  pinMode(ledPin, OUTPUT);
}

void loop() {
  // Allumer la LED
  digitalWrite(ledPin, HIGH);
  delay(2000); // Attendre 1 seconde

  // Éteindre la LED
  digitalWrite(ledPin, LOW);
  delay(2000); // Attendre 1 seconde
}
