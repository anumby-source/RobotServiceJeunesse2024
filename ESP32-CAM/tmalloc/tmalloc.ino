
#include <WiFi.h>

WiFiClient** serverClients = NULL;
int numberClients = 0;
WiFiClient* serverClient = NULL;

const char* ssid = "SSID";
const char* password = "123456789";


void addClient(WiFiClient* client) {
  WiFiClient** oldClients = serverClients;

  numberClients += 1;
  serverClients = (WiFiClient**) malloc(sizeof(WiFiClient*) * numberClients);

  if (numberClients > 1) {
    for (int i = 0; i < numberClients - 1; i++) {
      serverClients[i] = oldClients[i];
    }
  }
  serverClients[numberClients - 1] = client;
}

void showClients() {
  if (numberClients > 0) {
    for (int i = 0; i < numberClients; i++) {
      Serial.print((unsigned long) serverClients[i]);
      Serial.print(" - ");
    }
    Serial.println("");
  }
}



void setup() {
  Serial.begin(115200);

  IPAddress ESP8266_ip ( 192, 168, 0, 155);
  IPAddress dns_ip ( 192, 168, 0, 1);
  //IPAddress dns_ip ( 8, 8, 8, 8);
  //IPAddress dns_ip ( 192, 168, 0, 0);
  IPAddress gateway_ip ( 192, 168, 0, 1);
  IPAddress subnet_mask(255, 255, 255, 0);

  WiFi.config(ESP8266_ip, gateway_ip, subnet_mask);


  for (int i = 0; i < 5; i++) {
    addClient(serverClient);
    showClients();
  }
}

void loop() {
}

