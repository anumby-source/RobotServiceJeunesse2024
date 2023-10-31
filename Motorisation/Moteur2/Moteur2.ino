//Constants
#define nbL298N 2

//Parameters
const int ena1 = 2;
const int ena2 = 4;

const int fwd1 = 3;
const int bwd1 = 5;

const int fwd2 = 6;
const int bwd2 = 9;



//Variables
int Power = 400;
int D = 2000;

void setup() {
  //Init Serial USB
  Serial.begin(9600);
  Serial.println(F("Initialize System"));

  //Init DCmotor
  pinMode(ena1, OUTPUT);
  pinMode(fwd1, OUTPUT);
  pinMode(bwd1, OUTPUT);

  pinMode(ena2, OUTPUT);
  pinMode(fwd2, OUTPUT);
  pinMode(bwd2, OUTPUT);

}

void loop() {
  testL298N();
}

void testL298N() { /* function testL298N */
  ////Scenario to test H-Bridge
  //Forward
    digitalWrite(ena1, HIGH);
    digitalWrite(ena2, HIGH);

    analogWrite(bwd1, 0);
    analogWrite(bwd2, 0);

    for (int j = 200; j <= Power; j = j + 10) {
      Serial.println(j);
      analogWrite(fwd1, j);
      delay(100);
      analogWrite(fwd2, j);
      delay(D);
    }
    for (int j = Power; j >= 0; j = j - 10) {
      Serial.println(j);
      analogWrite(fwd1, j);
      delay(100);
      analogWrite(fwd2, j);
      delay(D);
    }
    delay(2000);


}
