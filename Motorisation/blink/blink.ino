// Inclure la bibliothèque ESP8266
#include <ESP8266WiFi.h>


// Broche à laquelle est connectée la LED (D0/GPIO 16 sur NodeMCU)
const int ledPin = 2; // D4   GPIO2    TXD1
const int ledP5 = 5; //  // D1 = GPIO5

#define led LED_BUILTIN
#define led 1 // TX = GPIO1
#define led 2 // D4 = GPIO2
#define led 3 // RX = GPIO3
#define led 4 // D2 = GPIO4
#define led 5 // D1 = GPIO5
#define led 12 // D6 = GPIO12
#define led 13 // D7 = GPIO13
#define led 14 // D5 = GPIO14
#define led 15 // D8 = GPIO15
#define led 16 // D0 = GPIO16


void setup() {
  // Initialisation de la broche de la LED en sortie
  pinMode(ledPin, OUTPUT);
  pinMode(5, OUTPUT);
  pinMode(14, OUTPUT);
}

void loop() {
  // Allumer la LED
  digitalWrite(ledPin, HIGH);
  digitalWrite(5, HIGH);
  digitalWrite(14, HIGH);
  delay(2000); // Attendre 1 seconde

  // Éteindre la LED
  digitalWrite(ledPin, LOW);
  digitalWrite(5, LOW);
  digitalWrite(14, LOW);
  delay(2000); // Attendre 1 seconde
}
