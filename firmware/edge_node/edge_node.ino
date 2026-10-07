#include <WiFi.h>
#include <HTTPClient.h>
#include <time.h>
#include "esp_sleep.h"

const char* WIFI_SSID = "YOUR_WIFI";
const char* WIFI_PASS = "YOUR_PASSWORD";
const char* SERVER_URL = "http://192.168.1.10:5000/api/telemetry";
const char* NODE_ID = "J3";

const int PRESSURE_PIN = 34;
const int RELAY_PIN = 26;
const int WAKE_PIN = 33;
const float SENSOR_MAX_KPA = 1000.0;
const float ADC_V_MIN = 0.33;
const float ADC_V_MAX = 2.97;
const float FAILSAFE_DROP_KPA = 80.0;
const int FAILSAFE_HOLD_SAMPLES = 10;
const uint64_t UPLOAD_INTERVAL_US = 30ULL * 1000000ULL;
const int BUFFER_LEN = 30;

RTC_DATA_ATTR float buffer[BUFFER_LEN];
RTC_DATA_ATTR time_t bufferTime[BUFFER_LEN];
RTC_DATA_ATTR int bufferCount = 0;
RTC_DATA_ATTR float referenceKpa = 0;
RTC_DATA_ATTR bool valveClosed = false;
RTC_DATA_ATTR bool cloudReachable = true;

float readPressureKpa() {
  long total = 0;
  for (int i = 0; i < 16; i++) {
    total += analogReadMilliVolts(PRESSURE_PIN);
  }
  float volts = (total / 16.0) / 1000.0;
  float span = (volts - ADC_V_MIN) / (ADC_V_MAX - ADC_V_MIN);
  return constrain(span, 0.0, 1.0) * SENSOR_MAX_KPA;
}

void updateReference(float kpa) {
  if (referenceKpa <= 0) {
    referenceKpa = kpa;
  } else {
    referenceKpa = 0.97 * referenceKpa + 0.03 * kpa;
  }
}

bool connectWifi() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  unsigned long start = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - start < 4000) {
    delay(100);
  }
  return WiFi.status() == WL_CONNECTED;
}

bool uploadBuffer(float latest) {
  if (!connectWifi()) {
    return false;
  }
  HTTPClient http;
  http.begin(SERVER_URL);
  http.addHeader("Content-Type", "application/json");
  time_t now = time(nullptr);
  String body = "[";
  for (int i = 0; i < bufferCount; i++) {
    long age = (long)(now - bufferTime[i]);
    body += "{\"node\":\"" + String(NODE_ID) + "\",\"pressure_kpa\":" + String(buffer[i], 2) + ",\"age_s\":" + String(age < 0 ? 0 : age) + "},";
  }
  body += "{\"node\":\"" + String(NODE_ID) + "\",\"pressure_kpa\":" + String(latest, 2) + "}]";
  int code = http.POST(body);
  http.end();
  WiFi.disconnect(true);
  WiFi.mode(WIFI_OFF);
  return code == 200;
}

void bufferReading(float kpa) {
  if (bufferCount == BUFFER_LEN) {
    for (int i = 1; i < BUFFER_LEN; i++) {
      buffer[i - 1] = buffer[i];
      bufferTime[i - 1] = bufferTime[i];
    }
    bufferCount--;
  }
  buffer[bufferCount] = kpa;
  bufferTime[bufferCount] = time(nullptr);
  bufferCount++;
}

void closeValve() {
  digitalWrite(RELAY_PIN, HIGH);
  valveClosed = true;
}

bool runFailsafeCheck() {
  int low = 0;
  for (int i = 0; i < FAILSAFE_HOLD_SAMPLES * 3; i++) {
    float kpa = readPressureKpa();
    if (referenceKpa - kpa > FAILSAFE_DROP_KPA) {
      low++;
      if (low >= FAILSAFE_HOLD_SAMPLES) {
        return true;
      }
    } else {
      low = 0;
    }
    delay(100);
  }
  return false;
}

void setup() {
  pinMode(RELAY_PIN, OUTPUT);
  digitalWrite(RELAY_PIN, valveClosed ? HIGH : LOW);
  pinMode(WAKE_PIN, INPUT);
  analogReadResolution(12);

  esp_sleep_wakeup_cause_t cause = esp_sleep_get_wakeup_cause();
  float kpa = readPressureKpa();

  if (cause == ESP_SLEEP_WAKEUP_EXT0) {
    bool burst = runFailsafeCheck();
    if (burst && !cloudReachable && !valveClosed) {
      closeValve();
    }
    float latest = readPressureKpa();
    cloudReachable = uploadBuffer(latest);
    if (cloudReachable) {
      bufferCount = 0;
    } else {
      bufferReading(latest);
    }
    if (burst && !cloudReachable && !valveClosed) {
      closeValve();
    }
    if (!burst) {
      updateReference(kpa);
    }
  } else {
    cloudReachable = uploadBuffer(kpa);
    if (cloudReachable) {
      bufferCount = 0;
    } else {
      bufferReading(kpa);
    }
    updateReference(kpa);
  }

  esp_sleep_enable_timer_wakeup(UPLOAD_INTERVAL_US);
  // ext0 is level triggered: re-arming it while the comparator is still high would wake straight away in a loop
  if (digitalRead(WAKE_PIN) == LOW) {
    esp_sleep_enable_ext0_wakeup((gpio_num_t)WAKE_PIN, 1);
  }
  esp_deep_sleep_start();
}

void loop() {
}
