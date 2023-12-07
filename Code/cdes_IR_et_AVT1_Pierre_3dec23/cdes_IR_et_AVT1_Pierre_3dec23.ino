// Equivalences entre la notation des pins du ESP2866 Nodemcu
//
//  référence: pinout description
// https://github.com/anumby-source/RobotServiceJeunesse2024/tree/main/ESP8266%20Mini%2CNodeMCU#pins-esp8266-nodemcu-v3
//
//   les broches D1 .. D8 .. D12 correspondent aux GPIO<n> qui peuvent être configurées au choix en INPUT ou OUTPUT

#include <IRremoteESP8266.h>
#include <IRrecv.h>
#include <IRutils.h>

const int receiverPin = D7; // Broche du récepteur IR
IRrecv irrecv(receiverPin);
decode_results results;

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

// definitions des commandes générées par une TCD IR, les codes en hexa associés sont spécifiques à une TCD
#define ARRET  FFA25D    //bouton rouge, 		      arrêt des moteurs, ENA ENB à 0
#define ACCEL  FF629D    //bouton VOL+,
#define FREIN  FFA857    //bouton VOL-, 
#define VIREG  FF22DD    //bouton retour rapide,  le DC du moteur G est diminué de 10 pendant 1s
#define VIRED  FFC23D    //bouton avance rapide,  le DC du moteur D est diminué de 10 pendant 1s
#define ARR1S  FFE01F    //bouton down, 		      marche arrière à Vmini pendant 0.3s
#define AVT1S  FF906F    //bouton up, 		        marche avant à Vmini pendant 0.3s
#define AVT1   FF30CF    //bouton 1, 			        marche avant à Vmini (DC= 700)
#define AVT2   FF18E7    //bouton 2, 			        marche avant à Vinter (DC= 850)
#define AVT3   FF7A85    //bouton 3, 			        marche avant à Vmax (DC= 1000)
#define ARR1   FF10EF    //bouton 4, 			        marche arr à Vmini (DC= 700)
#define ARR2   FF38C7    //bouton 5, 			        marche arr à Vinter (DC= 850)
#define ARR3   FF5AA5    //bouton 6, 			        marche arr à Vmax (DC= 1000)
#define SPAR1  FF42BD    //bouton 7,			        spare
#define SPAR2  FF4AB5    //bouton 8,			        spare
#define SPAR3  FF52AD    //bouton 9,			        spare
#define SPAR4  FF6897    //bouton 0,			        spare
#define SPAR5  FF02FD    //bouton avance/pause,	  spare
#define SPAR6  FF9867    //bouton EQ,			        spare
#define SPAR7  FFE21D    //bouton FUNC/STOP, 	    spare      
#define SPAR8  FFB04K    //bouton ST/REPT,		    spare


//Parameters
const int EnaG = D3;  // enable du moteur Gauche
const int EnaD = D4;  // enable du moteur Droit
const int fwd1 = D5;  // forward du moteur Gauche
const int bwd1 = D6;  // backward du moteur Gauche
const int fwd2 = D2;  // forward du moteur Droit
const int bwd2 = D1;  // backward du moteur Droit


//
int pins[] = {D1, D2, D3, D4, D5, D6, D7, D8};
const size_t n = sizeof(pins) / sizeof(int);

const int ledPin = 2; 

const int range = 1000;
const int freq = 40000; // 100 .. 40000

// int receiverPin = 13; // Signal de la télécommande
// IRrecv irrecv(receiverPin);
// decode_results results;

void translateIR() // Fonction reliant le signal à la fonction associée
// describing Remote IR codes
{
switch(results.value)
  {
  case 0xFFA25D: Serial.println("POWER         ARRET"); 
    analogWrite(D3, 0);           //  enaG  ENA
    analogWrite(D4, 0);           //  enaD  ENB
    digitalWrite(D5, HIGH);       //  fwdG  IN1
    digitalWrite(D6, LOW);        //  bwdG  IN2
    digitalWrite(D2, HIGH);       //  fwdD  IN3  
    digitalWrite(D1, LOW);        //  bwdD  IN4   
    delay(5);
    break;

  case 0xFFE21D: Serial.println("FUNC/STOP     SPAR7"); break;
  case 0xFF629D: Serial.println("VOL+          ACCEL"); break;
  case 0xFF22DD: Serial.println("retour rapide VIREG"); 
    analogWrite(D3, 700);         //  enaG  ENA
    analogWrite(D4, 750);         //  enaD  ENB survitesse motD
    digitalWrite(D5, HIGH);       //  fwdG  IN1
    digitalWrite(D6, LOW);        //  bwdG  IN2
    digitalWrite(D2, HIGH);       //  fwdD  IN3  
    digitalWrite(D1, LOW);        //  bwdD  IN4   
    delay(300);                   //  survitesse motD pendant 0.3s
    analogWrite(D3, 700);         //  enaG  ENA
    analogWrite(D4, 700);         //  enaD  ENB
    break;
  case 0xFF02FD: Serial.println("avance/pause   SPAR5"); break;
  case 0xFFC23D: Serial.println("Favance rapide VIRED" ); 
    analogWrite(D3, 750);         //  enaG  ENA survitesse motG
    analogWrite(D4, 700);         //  enaD  ENB
    digitalWrite(D5, HIGH);       //  fwdG  IN1
    digitalWrite(D6, LOW);        //  bwdG  IN2
    digitalWrite(D2, HIGH);       //  fwdD  IN3  
    digitalWrite(D1, LOW);        //  bwdD  IN4   
    delay(300);                   //  survitesse motG pendant 0.3s
    analogWrite(D3, 700);         //  enaG  ENA
    analogWrite(D4, 700);         //  enaD  ENB
    break;
  case 0xFFE01F: Serial.println("DOWN         ARR1S"); 
    analogWrite(D3, 700);         //  enaG  ENA
    analogWrite(D4, 700);         //  enaD  ENB
    digitalWrite(D5, LOW);        //  fwdG  IN1
    digitalWrite(D6, HIGH);       //  bwdG  IN2
    digitalWrite(D2, LOW);        //  fwdD  IN3  
    digitalWrite(D1, HIGH);       //  bwdD  IN4   
    delay(300);
    analogWrite(D3, 0);           //  enaG  ENA
    analogWrite(D4, 0);           //  enaD  ENB
    break;


  case 0xFFA857: Serial.println("VOL-        "); break;
  case 0xFF906F: Serial.println("UP           AVT1S"); 
    analogWrite(D3, 700);         //  enaG  ENA
    analogWrite(D4, 700);         //  enaD  ENB
    digitalWrite(D5, HIGH);       //  fwdG  IN1
    digitalWrite(D6, LOW);        //  bwdG  IN2
    digitalWrite(D2, HIGH);       //  fwdD  IN3  
    digitalWrite(D1, LOW);        //  bwdD  IN4   
    delay(300);
    analogWrite(D3, 0);           //  enaG  ENA
    analogWrite(D4, 0);           //  enaD  ENB
    break;


  case 0xFF9867: Serial.println("EQ           SPAR6"); break; 
  case 0xFFB04F: Serial.println("ST/REPT      SPAR8"); break;
  case 0xFF6897: Serial.println("0            SPAR4"); break;
  case 0xFF30CF: Serial.println("1            AVT1");
    analogWrite(D3, 700);         //  enaG  ENA avance à Vmin
    analogWrite(D4, 700);         //  enaD  ENB avance à Vmin
    digitalWrite(D5, HIGH);       //  fwdG  IN1
    digitalWrite(D6, LOW);        //  bwdG  IN2
    digitalWrite(D2, HIGH);       //  fwdD  IN3  
    digitalWrite(D1, LOW);        //  bwdD  IN4   
    delay(5);
    break;

  case 0xFF18E7: Serial.println("2            AVT2"); 
    analogWrite(D3, 850);         //  enaG  ENA  avance à Vinter
    analogWrite(D4, 850);         //  enaD  ENB  avance à Vinter
    digitalWrite(D5, HIGH);       //  fwdG  IN1
    digitalWrite(D6, LOW);        //  bwdG  IN2
    digitalWrite(D2, HIGH);       //  fwdD  IN3  
    digitalWrite(D1, LOW);        //  bwdD  IN4   
    delay(5);
    break;


  case 0xFF7A85: Serial.println("3            AVT3");
    analogWrite(D3, 1000);        //  enaG  ENA avance à Vmax
    analogWrite(D4, 1000);        //  enaD  ENB avance à Vmax
    digitalWrite(D5, HIGH);       //  fwdG  IN1
    digitalWrite(D6, LOW);        //  bwdG  IN2
    digitalWrite(D2, HIGH);       //  fwdD  IN3  
    digitalWrite(D1, LOW);        //  bwdD  IN4   
    delay(5);
    break;
  case 0xFF10EF: Serial.println("4            ARR1"); // marche arrière permanente à Vmin
    analogWrite(D3, 700);         //  enaG  ENA arrière à Vmin
    analogWrite(D4, 700);         //  enaD  ENB arrière à Vmin
    digitalWrite(D5, LOW);        //  fwdG  IN1
    digitalWrite(D6, HIGH);       //  bwdG  IN2
    digitalWrite(D2, LOW);        //  fwdD  IN3  
    digitalWrite(D1, HIGH);       //  bwdD  IN4   
    delay(500);
    break;


  case 0xFF38C7: Serial.println("5            ARR2"); // marche arrière permanente à Vinter
    analogWrite(D3, 850);         //  enaG  ENA arrière à Vinter
    analogWrite(D4, 850);         //  enaD  ENB arrière à Vinter
    digitalWrite(D5, LOW);        //  fwdG  IN1
    digitalWrite(D6, HIGH);       //  bwdG  IN2
    digitalWrite(D2, LOW);        //  fwdD  IN3  
    digitalWrite(D1, HIGH);       //  bwdD  IN4   
    delay(500);
    break;


  case 0xFF5AA5: Serial.println("6            ARR3"); // marche arrière permanente à Vmax
    analogWrite(D3, 1000);        //  enaG  ENA arrière à Vmax
    analogWrite(D4, 1000);        //  enaD  ENB arrière à Vmax
    digitalWrite(D5, LOW);        //  fwdG  IN1
    digitalWrite(D6, HIGH);       //  bwdG  IN2
    digitalWrite(D2, LOW);        //  fwdD  IN3  
    digitalWrite(D1, HIGH);       //  bwdD  IN4   
    delay(500);
    break;


  case 0xFF42BD: Serial.println("7            SPAR1"); break;
  case 0xFF4AB5: Serial.println("8            SPAR2"); break;
  case 0xFF52AD: Serial.println("9            SPAR3"); break;
  case 0xFFFFFFFF: Serial.println(" Répéter");break;
  default:
  Serial.println(" Autre bouton ");
  }
  delay(500); // Permet de laisser le temps de recevoir le prochain signal
}



void setup() {
  analogWriteRange(range);
  analogWriteFreq(freq);
  irrecv.enableIRIn(); // Initialise le récepteur IR
  // Initialisez ici la configuration de votre L298N et de vos moteurs

  //Init Serial USB
  Serial.begin(9600);
  Serial.println(F("Initialize System"));

  //Init DCmotor
  pinMode(EnaG, OUTPUT);
  pinMode(fwd1, OUTPUT);
  pinMode(bwd1, OUTPUT);

  pinMode(EnaD, OUTPUT);
  pinMode(fwd2, OUTPUT);
  pinMode(bwd2, OUTPUT);
}




void loop() // Boucle qui s'éxécute à l'infini
{
  if (irrecv.decode(&results)) 
  {
    translateIR();
    irrecv.resume(); // Permet de recevoir la valeur suivante
  }
}
