#include <IRremote.hpp>

#include <Arduino.h>
#include <IRremote.h>

#define GND 5  // D1
#define IN  0  // ADC0
#define IN  4  // D2
#define led 12 // D6

IRrecv irrecv(IN);


decode_results results;

int i = 0;

void setup()
{
  Serial.begin(115200);
  Serial.println("Setup....");
  pinMode(GND, OUTPUT);      // board IR
  pinMode(led, OUTPUT);

  digitalWrite(GND, LOW);
  
  IrReceiver.begin(IR_RECEIVE_PIN, ENABLE_LED_FEEDBACK, USE_DEFAULT_FEEDBACK_LED_PIN);
    Serial.print(F("Ready to receive IR signals at pin "));
    Serial.println(IR_RECEIVE_PIN);  
 irrecv.enableIRIn();  // Start the receiver
  while (!Serial)  // Wait for the serial connection to be establised.
    delay(50);
  Serial.println();
  Serial.print("IRrecv is now running and waiting for IR message on Pin ");
}

void loop() {
  if (irrecv.decode(&results)) {
    Serial.println(results.value, HEX);
    irrecv.resume();
  }
}