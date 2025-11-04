# app/kafka_consumer.py
from kafka import KafkaConsumer
import json
import torch
from services.online_trainer import OnlineTrainer
from pathlib import Path

trainer = OnlineTrainer()
consumer = KafkaConsumer(
    "nimway-sensors",
    bootstrap_servers=["localhost:9092"],
    value_deserializer=lambda x: json.loads(x.decode("utf-8")),
    auto_offset_reset="earliest",
    enable_auto_commit=True,
    group_id="vae-consumer",
)

print("Listening...")
for msg in consumer:
    data = msg.value
    rid = data["resourceid"]
    seq_id = data["seq_id"]

    flat = [
        data["sensors"][f"{c}_t{t}"]
        for t in range(10)
        for c in [
            "temperature_value",
            "humidity_value",
            "co2_value",
            "light_value",
            "radon_value",
            "airquality_value",
        ]
    ]
    seq = torch.tensor(flat, dtype=torch.float32).unsqueeze(0)

    error, is_anomaly, thresh = trainer.update(seq, rid)

    if is_anomaly:
        anomaly = {
            "seq_id": seq_id,
            "resourceid": rid,
            "timestamp": data["start_time"],
            "error": round(error, 3),
            "threshold": round(thresh, 3),
        }
        Path("data/current_anomaly.json").write_text(json.dumps(anomaly))
        print(f"ALERT → {anomaly}")
    else:
        print(f"Normal: {error:.3f}")
