#include "WiFi.h"
#include "ESPAsyncWebSrv.h"

const char* ssid = "ESP32-CAMWeb-server";
const char* password =  "123456789";

AsyncWebServer server1(80);
AsyncWebServer server2(81);
AsyncWebServer server3(82);

void setup(){
  Serial.begin(115200);

  WiFi.mode(WIFI_AP);
  WiFi.softAP(ssid, password);

  //while (1) {
  //  int status = WiFi.status();
  //  Serial.print("status = ");
  //  Serial.print(status);
  //  Serial.println(" ... Connecting to WiFi..");
  //  if (status == WL_CONNECTED) break;
  //  delay(1000);
  //}

  IPAddress IP = WiFi.softAPIP();
  Serial.print("AP IP address: ");
  Serial.println(IP);
  Serial.println(WiFi.localIP());

  server1.on("/hello", HTTP_GET, [](AsyncWebServerRequest *request){
    request->send(200, "text/plain", "Hello from server 1");
  });

  server3.on("/test", HTTP_GET, [](AsyncWebServerRequest *request){
    request->send(200, "text/plain", "Hello from server 3");
  });

  server1.begin();
  server2.begin();
  server3.begin();
}

void loop(){
  delay(1000);

  server2.on("/hello", HTTP_GET, [](AsyncWebServerRequest *request){
    for (int i = 0; i < 20; i++) {
      request->send(200, "text/plain", "Hello from server 2 <br>");
    }
  });

}
