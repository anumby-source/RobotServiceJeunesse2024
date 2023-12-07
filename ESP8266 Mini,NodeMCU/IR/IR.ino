// Equivalences entre la notation des pins du ESP2866 Nodemcu
//
//  référence: pinout description
// https://github.com/anumby-source/RobotServiceJeunesse2024/tree/main/ESP8266%20Mini%2CNodeMCU#pins-esp8266-nodemcu-v3
//
//   les broches D1 .. D8 .. D12 correspondent aux GPIO<n> qui peuvent être configurées au choix en INPUT ou OUTPUT

#define D0 16  // D0 = GPIO16

#define D1 5   // D1 = GPIO5
#define D2 4   // D2 = GPIO4
#define D3 0   // D3 = GPIO0
#define D4 2   // D4 = GPIO2 (internal led)

#define P3V3_1 0
#define GND_1  0

#define D5 14  // D5 = GPIO14
#define D6 12  // D6 = GPIO12
#define D7 13  // D7 = GPIO13
#define D8 15  // D8 = GPIO15

#define D9  3   // D9 = GPIO3
#define D10 1   // D10 = GPIO1
#define D11 9   // D11 = GPIO9
#define D12 10  // D12 = GPI10

#define RX 3   // RX = GPIO3
#define TX 1   // TX = GPIO1

#define P3V3_2 0
#define GND_2  0

#define ADC    0
#define GND_3  0
#define VU     0

#define S3 10  // S3 = GPIO10      ESP8266 Nodemcu
#define S2 9   // S2 = GPIO9       ESP8266 Nodemcu
#define S1 8   // S1 = GPIO8       ESP8266 Nodemcu
#define SC 11  // SC = GPIO11      ESP8266 Nodemcu
#define S0 7   // S0 = GPIO7       ESP8266 Nodemcu
#define SK 6   // SK = GPIO6       ESP8266 Nodemcu

#define GND_4  0
#define GND_4  0



//
int pins[] = {D1, D2, D3, D4, D5, D6, D7, D8};
const size_t n = sizeof(pins) / sizeof(int);

volatile byte state = LOW;
const byte ledPin = D4;
const byte IRPin = D5; 
int buttonPin = D5; // le bouton = pin D5
const int mode = CHANGE;   // CHANGE or FALLING or RISING
bool buttonState = false; 

ICACHE_RAM_ATTR void blink() {
  if (state == LOW) state = HIGH;
  else state = LOW;
}

void setup() {
  Serial.begin(115200);
  pinMode(ledPin, OUTPUT);
  pinMode(buttonPin, INPUT_PULLUP);
  // pinMode(IRPin, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(buttonPin), blink, mode);
}

void loop() {
  //delay(1000);
  //blink();
  //digitalWrite(ledPin, state);
  // condition 1 : bouton appuyé ET état actuel du bouton éteint
  buttonState = digitalRead(buttonPin);
  Serial.println(buttonState);
  if (buttonState == HIGH)
  {
    digitalWrite(ledPin, HIGH); // j'allume ma led 
  }
  else
  {
    digitalWrite(ledPin, LOW); // j'éteins ma led
  }
  delay(1000);
}
