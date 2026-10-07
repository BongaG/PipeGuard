import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "pipeguard-dev")
    DATABASE = os.path.join(BASE_DIR, "instance", "pipeguard.db")
    MODEL_DIR = os.path.join(BASE_DIR, "models")
    RESULTS_DIR = os.path.join(BASE_DIR, "results")
    SAMPLE_HZ = 1
    WINDOW_S = 10
    BASELINE_S = 60
    ALARM_CONSECUTIVE = 2
    SCHEDULED_UPLOAD_S = 30
    ULP_DEVIATION_KPA = 4.0
    ULP_RATE_KPA_S = 3.0
    ULP_REFRACTORY_S = 30
    DEMAND_EVENT_S = 60
    EVENT_MIN_CUT = 0.02
    LATCH_OFFSET_S = (-8, 1)
    FAILSAFE_DROP_KPA = 80.0
    FAILSAFE_HOLD_SAMPLES = 10
    FAILSAFE_SAMPLE_HZ = 10
    VALVE_CLOSE_S = (1.0, 2.0)
    LORA_LATENCY_S = (0.6, 1.8)
    GSM_LATENCY_S = (1.5, 4.0)
    LORA_LOSS = 0.05
    SENSOR_NOISE_KPA = 0.5
    SENSOR_DRIFT_KPA = 2.0
    FLOW_NOISE_FRAC = 0.01
    ACTIVE_MA = 70.0
    ACTIVE_WAKE_S = 1.2
    SLEEP_ULP_MA = 0.15
    ALWAYS_ON_MA = 80.0
    BATTERY_MAH = 3000.0
    TICK_S = float(os.environ.get("PIPEGUARD_TICK_S", "1.0"))
    BUFFER_S = 600
    AUTO_ISOLATE_BURSTS = True
    AUTO_ISOLATE_CONFIDENCE = 0.8
    FAILSAFE_ONLY_OFFLINE = True
    ALERT_CLEAR_S = 30
    TELEMETRY_OVERRIDE_S = 3.0
    LEAK_PERCENT = 25.0
    BURST_PERCENTS = (50.0, 100.0)
    START_ENGINE = True
