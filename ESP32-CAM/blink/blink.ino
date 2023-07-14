
const int flashPin = 4;

void setup(){
  //setup the pint for the flash
  pinMode(flashPin, OUTPUT);
  digitalWrite(flashPin, LOW);
}

void loop(){
  delay(500);
  digitalWrite(flashPin, HIGH);
  delay(500);
  digitalWrite(flashPin, LOW);
}
