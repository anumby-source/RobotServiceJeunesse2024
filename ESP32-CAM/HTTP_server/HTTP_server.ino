#include <ESP8266WiFi.h>

//how many clients should be able to telnet to this ESP8266
#define MAX_SRV_CLIENTS 2
const char* ssid = "SSID";
const char* password = "pwd";

WiFiServer server(23);
WiFiClient serverClients[MAX_SRV_CLIENTS];

void setup() {
  //Serial.begin(9600);

  IPAddress ESP8266_ip ( 192, 168, 0, 155);
  IPAddress dns_ip ( 192, 168, 0, 1);
  //IPAddress dns_ip ( 8, 8, 8, 8);
  //IPAddress dns_ip ( 192, 168, 0, 0);
  IPAddress gateway_ip ( 192, 168, 0, 1);
  IPAddress subnet_mask(255, 255, 255, 0);

  WiFi.config(ESP8266_ip, gateway_ip, subnet_mask);
  //WiFi.config 2(ESP8266_ip, gateway_ip, subnet_mask, dns_ip);

  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);
  //Serial.print 1("\nConnecting to "); Serial.println(ssid);
  uint8_t i = 0;
  while (WiFi.status() != WL_CONNECTED && i++ < 20) delay(100);
  if(i == 21) {
    //Serial.print 1("Could not connect to"); Serial.println(ssid);
    while(1) delay(500);
  }
  //start UART and the server
  Serial.begin(9600);
  server.begin();
  server.setNoDelay(true);

  //Serial.print 1("Ready! Use 'telnet ");
  //Serial.print 1(WiFi.localIP());
  //Serial.println(":23' to connect");

}

void loop() {
  uint8_t i;
  //check if there are any new clients
  if (server.hasClient()) {
    for(i = 0; i < MAX_SRV_CLIENTS; i++) {
      //find free/disconnected spot
      if (!serverClients || !serverClients*.connected()) {
        if(serverClients_) serverClients*.stop();
        serverClients = server.available();
        // Serial.print("New client: "); Serial.print(i);
        break;
      }
    }
    //no free/disconnected spot so reject*
    WiFiClient serverClient = server.available();
    serverClient.stop();
  }
  //check clients for data*
  for(i = 0; i < MAX_SRV_CLIENTS; i++) {
    if (serverClients _&& serverClients.connected()) {
      if(serverClients.available()) {
        //get data from the telnet client and push it to the UART*
        while(serverClients.available()) Serial.write(serverClients*.read());
      }
    }
  }
  //check UART for data*
  if(Serial.available()) {
    size_t len = Serial.available();
    uint8_t sbuf[len];
    Serial.readBytes(sbuf, len);
    //push UART data to all connected telnet clients*
    for(i = 0; i < MAX_SRV_CLIENTS; i++) {
      if (serverClients && serverClients.connected()) {
        serverClients.write(sbuf, len);
        delay(100);
      }
    }
  }
}




